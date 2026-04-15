import os, sys, time, pathlib, warnings, xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

DELAY_BETWEEN_CHUNKS = 6

ALLOWED_RELATIONS = [
    "can_treat_disease", "may_suffer_from_disease", "should_eat",
    "common_medication_is", "accompanied_by_symptoms_of",
    "should_avoid_eating", "symptoms_are", "diagnostic_tests",
]

# ── Model with fallback chain ──────────────────────────────────────────────────
# Primary: llama-4-scout (30K TPM — best for big chunks)
# Fallback: qwen3-32b (60 RPM), then llama-3.1-8b-instant (14.4K RPD)
MODEL_CHAIN = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]

def make_llm(model: str):
    return ChatGroq(model=model, temperature=0, api_key=os.getenv("GROQ_API_KEY"))

def groq_invoke_with_retry(chain_fn, inputs: dict, max_wait: int = 300):
    """
    Retries forever (up to max_wait seconds total sleep) on ANY error.
    Uses exponential backoff. Cycles through model fallbacks on rate limit.
    """
    model_idx = 0
    wait = 10  # start with 10s

    while True:
        model = MODEL_CHAIN[model_idx % len(MODEL_CHAIN)]
        try:
            return chain_fn(model, inputs)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in ["429", "rate limit", "too many", "quota"])

            if is_rate_limit:
                print(f"  [rate limit] model={model} | sleeping {wait}s then trying next model...")
                time.sleep(wait)
                wait = min(wait * 2, 120)  # cap at 2 min
                model_idx += 1             # rotate model
            else:
                print(f"  [error] {e} | sleeping {wait}s and retrying...")
                time.sleep(wait)
                wait = min(wait * 2, 120)

# ── LLM chain builders ─────────────────────────────────────────────────────────

JARGON_DETECT_PROMPT = PromptTemplate.from_template("""
Read the following medical text and list any abbreviations or domain-specific
jargon a general reader might not understand.
Output a comma-separated list only. If none, output: NONE

Text: {text}
""")

JARGON_RESOLVE_PROMPT = PromptTemplate.from_template("""
You are a medical expert.
Search result context: {search_result}
Explain what '{term}' means in the medical field in 1 sentence.
If unknown, say: Unknown term.
""")

TRIPLE_EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are a medical knowledge graph expert.
Extract as many knowledge triples as possible from the text below.

Format — one triple per line:
head_entity | relation | tail_entity

Allowed relations (use exactly as written, no others):
{allowed_relations}

Rules:
- Entities: concise lowercase English noun phrases
- Only use relations from the allowed list
- Use resolved forms for any jargon (see below)
- Output ONLY the triples — no headers, no explanations, no numbering

Resolved jargon:
{resolved_jargon}

Text:
{text}
""")

def _jargon_detect(model, inputs):
    chain = JARGON_DETECT_PROMPT | make_llm(model) | StrOutputParser()
    return chain.invoke(inputs)

def _jargon_resolve(model, inputs):
    chain = JARGON_RESOLVE_PROMPT | make_llm(model) | StrOutputParser()
    return chain.invoke(inputs)

def _triple_extract(model, inputs):
    chain = TRIPLE_EXTRACTION_PROMPT | make_llm(model) | StrOutputParser()
    return chain.invoke(inputs)

# ── Neo4j ──────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_indexes()

    def close(self):
        self.driver.close()

    def _create_indexes(self):
        with self.driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    def insert_triple(self, head, relation, tail):
        rel_label = relation.upper().replace(" ", "_").replace("-", "_")
        query = f"""
        MERGE (h:Entity {{name: $head}})
        MERGE (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_label}]->(t)
        """
        with self.driver.session() as session:
            session.run(query, head=head.strip().lower(), tail=tail.strip().lower())

    def get_stats(self):
        with self.driver.session() as session:
            triples  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            entities = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            rel_types = [r["t"] for r in session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t").data()]
        return {"triples": triples, "entities": entities, "relation_types": rel_types}

    def get_questions_by_dataset(self, dataset_tag: str, limit: int = 20) -> list[str]:
        """Pull questions stored with a dataset tag from the KG."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (q:Question {dataset: $tag})
                RETURN q.text AS text
                LIMIT $limit
            """, tag=dataset_tag, limit=limit)
            return [r["text"] for r in result]

    def insert_question(self, question: str, dataset_tag: str):
        """Store source question in KG for later eval retrieval."""
        with self.driver.session() as session:
            session.run("""
                MERGE (q:Question {text: $text})
                SET q.dataset = $tag
            """, text=question.strip(), tag=dataset_tag)

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_mts_dialog(filepath, max_rows=None):
    df = pd.read_csv(filepath, encoding="utf-8")
    if max_rows:
        df = df.head(max_rows)
    chunks, questions = [], []
    for _, row in df.iterrows():
        section  = str(row.get("section_text", "")).strip()
        dialogue = str(row.get("dialogue", "")).strip()
        combined = " ".join(filter(None, [section, dialogue])).strip()
        if combined:
            chunks.append(combined)
            # First sentence of dialogue as question
            q = dialogue.split(".")[0].strip()
            if q:
                questions.append(q)
    print(f"  Loaded {len(chunks)} chunks from {filepath}")
    return chunks, questions

def load_medquad_xml(filepath):
    chunks, questions = [], []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for qapair in root.findall(".//QAPair"):
            q_el = qapair.find("Question")
            a_el = qapair.find("Answer")
            question = q_el.text.strip() if q_el is not None and q_el.text else ""
            answer   = a_el.text.strip()  if a_el is not None and a_el.text   else ""
            combined = " ".join(filter(None, [question, answer])).strip()
            if combined:
                chunks.append(combined)
            if question:
                questions.append(question)
    except ET.ParseError as e:
        print(f"  [XML ERROR] {filepath}: {e}")
    return chunks, questions

def load_medquad_folders(root_path, folder_names, max_files_per_folder=None):
    root = pathlib.Path(root_path)
    folders = [root / name for name in folder_names] if folder_names else [f for f in root.iterdir() if f.is_dir()]
    all_chunks, all_questions = [], []
    for folder in folders:
        if not folder.exists():
            print(f"  [WARN] Not found: {folder}")
            continue
        xml_files = sorted(folder.glob("*.xml"))
        if max_files_per_folder:
            xml_files = xml_files[:max_files_per_folder]
        for xml_file in xml_files:
            c, q = load_medquad_xml(xml_file)
            all_chunks.extend(c)
            all_questions.extend(q)
    print(f"  Total MedQuAD chunks: {len(all_chunks)}, questions: {len(all_questions)}")
    return all_chunks, all_questions

# ── DuckDuckGo ─────────────────────────────────────────────────────────────────

def search_term(term):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(f"{term} meaning medical field", max_results=2)
            if results:
                return results[0].get("body", "")
    except Exception:
        pass
    return ""

# ── Processing ─────────────────────────────────────────────────────────────────

def parse_triples(raw):
    triples = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            head, rel, tail = parts
            rel_clean = rel.lower().replace(" ", "_").replace("-", "_")
            if rel_clean in ALLOWED_RELATIONS:
                triples.append((head, rel_clean, tail))
    return triples

def process_chunk(text, kg):
    text = text[:2000]

    raw_jargon = groq_invoke_with_retry(_jargon_detect, {"text": text})
    jargon_terms = [] if raw_jargon.strip().upper() == "NONE" else [t.strip() for t in raw_jargon.split(",") if t.strip()]

    resolved = {}
    for term in jargon_terms:
        explanation = groq_invoke_with_retry(_jargon_resolve, {
            "term": term,
            "search_result": search_term(term),
        })
        resolved[term] = explanation.strip()
        print(f"    [jargon] {term} → {explanation[:70]}...")

    resolved_str = "\n".join(f"- {k}: {v}" for k, v in resolved.items()) or "None"

    raw = groq_invoke_with_retry(_triple_extract, {
        "text": text,
        "allowed_relations": "\n".join(f"- {r}" for r in ALLOWED_RELATIONS),
        "resolved_jargon": resolved_str,
    })

    triples = parse_triples(raw)
    print(f"  {len(triples)} triples extracted")
    for head, rel, tail in triples:
        kg.insert_triple(head, rel, tail)
        print(f"    + ({head}) -[{rel}]-> ({tail})")

    return len(triples)

def run_dataset(name, chunks, questions, kg, dataset_tag, delay=DELAY_BETWEEN_CHUNKS):
    print(f"\n{'='*60}\nDataset : {name} | Chunks: {len(chunks)}\n{'='*60}")

    # Store questions in KG for eval retrieval
    for q in questions[:50]:  # store up to 50 per dataset
        kg.insert_question(q, dataset_tag)
    print(f"  Stored {min(len(questions),50)} questions with tag='{dataset_tag}'")

    total = 0
    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk[:80].strip()}...")
        try:
            total += process_chunk(chunk, kg)
        except Exception as e:
            print(f"  [FATAL] {e}")
        if i < len(chunks) - 1:
            time.sleep(delay)

    print(f"\n{name} done — {total} triples inserted")
    return total

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print("Connected.\n")

    grand_total = 0

    print("\nLoading MTS-Dialog...")
    mts_chunks, mts_questions = load_mts_dialog("data/MTS-Dialog-TrainingSet.csv", max_rows=None)
    grand_total += run_dataset("MTS-Dialog", mts_chunks, mts_questions, kg, dataset_tag="MTS-Dialog")

    print("\nLoading MedQuAD...")
    mq_chunks, mq_questions = load_medquad_folders(
        root_path="data/MedQuAD",
        folder_names=["1_CancerGov_QA", "4_MPlus_Health_Topics_QA", "9_CDC_QA"],
        max_files_per_folder=20,
    )
    grand_total += run_dataset("MedQuAD", mq_chunks, mq_questions, kg, dataset_tag="MedQuAD")

    stats = kg.get_stats()
    print(f"\n{'='*60}\nFINAL STATS\n  Triples  : {stats['triples']}\n  Entities : {stats['entities']}\n  Rel types: {stats['relation_types']}\n{'='*60}")
    kg.close()