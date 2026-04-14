"""
MI-RAG: Knowledge Graph Construction
Datasets : MTS-Dialog (CSV) + MedQuAD (XML folders)
LLM      : openai/gpt-oss-120b via Groq (free tier)
Features : Rate limit handling, delay between chunks, XML + CSV loaders
"""

import os
import sys
import time
import pathlib
import warnings
import xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Delay between chunks to avoid Groq rate limits (seconds)
# Groq free tier = 30 req/min → 1 chunk uses ~3 LLM calls → safe at 6s delay
DELAY_BETWEEN_CHUNKS = 6

ALLOWED_RELATIONS = [
    "can_treat_disease",
    "may_suffer_from_disease",
    "should_eat",
    "common_medication_is",
    "accompanied_by_symptoms_of",
    "should_avoid_eating",
    "symptoms_are",
    "diagnostic_tests",
]

# ── LLM ────────────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Neo4j ──────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_indexes()

    def close(self):
        self.driver.close()

    def _create_indexes(self):
        with self.driver.session() as session:
            session.run(
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )

    def insert_triple(self, head: str, relation: str, tail: str):
        relation_label = relation.upper().replace(" ", "_").replace("-", "_")
        query = f"""
        MERGE (h:Entity {{name: $head}})
        MERGE (t:Entity {{name: $tail}})
        MERGE (h)-[r:{relation_label}]->(t)
        """
        with self.driver.session() as session:
            session.run(
                query,
                head=head.strip().lower(),
                tail=tail.strip().lower(),
            )

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            triples   = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            entities  = session.run("MATCH (e:Entity) RETURN count(e) AS c").single()["c"]
            rel_types = [
                r["t"] for r in
                session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t").data()
            ]
        return {"triples": triples, "entities": entities, "relation_types": rel_types}

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_mts_dialog(filepath: str, max_rows: int = None) -> list:
    """
    MTS-Dialog CSV loader.
    Combines section_text + dialogue columns into one chunk per row.
    """
    df = pd.read_csv(filepath, encoding="utf-8")

    for col in ["section_text", "dialogue"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Got: {list(df.columns)}")

    if max_rows:
        df = df.head(max_rows)

    chunks = []
    for _, row in df.iterrows():
        section  = str(row["section_text"]).strip() if pd.notna(row["section_text"]) else ""
        dialogue = str(row["dialogue"]).strip()     if pd.notna(row["dialogue"])     else ""
        combined = " ".join(filter(None, [section, dialogue])).strip()
        if combined:
            chunks.append(combined)

    print(f"  Loaded {len(chunks)} chunks from {filepath}")
    return chunks


def load_medquad_xml(filepath: str) -> list:
    """
    MedQuAD XML loader.
    Extracts Question + Answer text from each QAPair and combines them.
    Handles the structure:
      <Document>
        <QAPairs>
          <QAPair>
            <Question>...</Question>
            <Answer>...</Answer>
          </QAPair>
        </QAPairs>
      </Document>
    """
    chunks = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for qapair in root.findall(".//QAPair"):
            question_el = qapair.find("Question")
            answer_el   = qapair.find("Answer")

            question = question_el.text.strip() if question_el is not None and question_el.text else ""
            answer   = answer_el.text.strip()   if answer_el   is not None and answer_el.text   else ""

            combined = " ".join(filter(None, [question, answer])).strip()
            if combined:
                chunks.append(combined)

    except ET.ParseError as e:
        print(f"  [XML ERROR] Could not parse {filepath}: {e}")

    return chunks


def load_medquad_folder(folder_path: str, max_files: int = None) -> list:
    """
    Loads all XML files from a MedQuAD folder (e.g. 1_CancerGov_QA).
    Returns a flat list of all QA chunks across all files.
    """
    folder = pathlib.Path(folder_path)
    xml_files = sorted(folder.glob("*.xml"))

    if max_files:
        xml_files = xml_files[:max_files]

    print(f"  Found {len(xml_files)} XML files in {folder.name}")

    all_chunks = []
    for xml_file in xml_files:
        chunks = load_medquad_xml(xml_file)
        all_chunks.extend(chunks)

    print(f"  Loaded {len(all_chunks)} chunks from {folder.name}")
    return all_chunks


def load_medquad_folders(root_path: str, folder_names: list, max_files_per_folder: int = None) -> list:
    """
    Loads multiple MedQuAD folders at once.
    Pass folder_names=None to load ALL 12 folders.

    Example:
        load_medquad_folders("data/MedQuAD", ["1_CancerGov_QA", "4_MPlus_Health_Topics_QA"])
    """
    root = pathlib.Path(root_path)

    if folder_names is None:
        folders = [f for f in root.iterdir() if f.is_dir()]
    else:
        folders = [root / name for name in folder_names]

    all_chunks = []
    for folder in folders:
        if folder.exists():
            chunks = load_medquad_folder(str(folder), max_files=max_files_per_folder)
            all_chunks.extend(chunks)
        else:
            print(f"  [WARN] Folder not found: {folder}")

    print(f"\n  Total MedQuAD chunks loaded: {len(all_chunks)}")
    return all_chunks

# ── DuckDuckGo ─────────────────────────────────────────────────────────────────

def search_term(term: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(f"{term} meaning medical field", max_results=2)
            if results:
                return results[0].get("body", "")
    except Exception:
        pass
    return ""

# ── Prompts ────────────────────────────────────────────────────────────────────

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

jargon_detect_chain  = JARGON_DETECT_PROMPT     | llm | StrOutputParser()
jargon_resolve_chain = JARGON_RESOLVE_PROMPT    | llm | StrOutputParser()
triple_chain         = TRIPLE_EXTRACTION_PROMPT | llm | StrOutputParser()

# ── Processing ─────────────────────────────────────────────────────────────────

def extract_jargon(text: str) -> list:
    result = jargon_detect_chain.invoke({"text": text})
    if result.strip().upper() == "NONE":
        return []
    return [t.strip() for t in result.split(",") if t.strip()]

def resolve_jargon(terms: list) -> dict:
    resolved = {}
    for term in terms:
        explanation = jargon_resolve_chain.invoke({
            "term": term,
            "search_result": search_term(term),
        })
        resolved[term] = explanation.strip()
        print(f"    [jargon] {term} → {explanation[:70]}...")
    return resolved

def parse_triples(raw: str) -> list:
    triples = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            head, rel, tail = parts
            rel_clean = rel.lower().replace(" ", "_").replace("-", "_")
            if rel_clean in ALLOWED_RELATIONS:
                triples.append((head, rel_clean, tail))
    return triples

def process_chunk(text: str, kg: KnowledgeGraph) -> int:
    # Truncate very long chunks to avoid token limits
    text = text[:2000]

    jargon_terms = extract_jargon(text)
    if jargon_terms:
        print(f"  Jargon: {jargon_terms}")

    resolved     = resolve_jargon(jargon_terms) if jargon_terms else {}
    resolved_str = "\n".join(f"- {k}: {v}" for k, v in resolved.items()) or "None"

    raw = triple_chain.invoke({
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


def run_dataset(name: str, chunks: list, kg: KnowledgeGraph, delay: float = DELAY_BETWEEN_CHUNKS):
    """Process a list of chunks into the KG with rate limit delay."""
    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"Chunks  : {len(chunks)}")
    print(f"Delay   : {delay}s between chunks")
    print(f"{'='*60}")

    total = 0
    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk[:80].strip()}...")
        try:
            total += process_chunk(chunk, kg)
        except Exception as e:
            print(f"  [ERROR] Skipping chunk: {e}")

        # Rate limit guard — sleep between chunks
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

    # ── DATASET 1: MTS-Dialog CSV ─────────────────────────────────────────────
    # Remove max_rows once you're confident the pipeline works
    print("\nLoading MTS-Dialog...")
    mts_chunks = load_mts_dialog(
        "data/MTS-Dialog-TrainingSet.csv",
        max_rows=None,       # ← set to e.g. 50 for quick test, None for full
    )
    grand_total += run_dataset("MTS-Dialog", mts_chunks, kg)

    # ── DATASET 2: MedQuAD XML ────────────────────────────────────────────────
    # Recommended: start with 3 folders. Add more once you're happy with results.
    # Available folders:
    #   1_CancerGov_QA, 2_GARD_QA, 3_GHR_QA, 4_MPlus_Health_Topics_QA,
    #   5_NIDDK_QA, 6_NINDS_QA, 7_SeniorHealth_QA, 8_NHLBI_QA_XML,
    #   9_CDC_QA, 10_MPlus_ADAM_QA, 11_MPlusDrugs_QA, 12_MPlusHerbsSupplements_QA
    print("\nLoading MedQuAD...")
    medquad_chunks = load_medquad_folders(
        root_path="data/MedQuAD",
        folder_names=[
            "1_CancerGov_QA",
            "4_MPlus_Health_Topics_QA",
            "9_CDC_QA",
        ],
        max_files_per_folder=20,  # ← remove or increase once tested
    )
    grand_total += run_dataset("MedQuAD", medquad_chunks, kg)

    # ── Final stats ───────────────────────────────────────────────────────────
    stats = kg.get_stats()
    print(f"\n{'='*60}")
    print(f"FINAL STATS")
    print(f"  Total triples   : {stats['triples']}")
    print(f"  Total entities  : {stats['entities']}")
    print(f"  Relation types  : {stats['relation_types']}")
    print(f"{'='*60}")

    kg.close()