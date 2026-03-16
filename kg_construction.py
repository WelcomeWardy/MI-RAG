"""
MI-RAG: Knowledge Graph Construction
Dataset : MTS-Dialog (CSV columns: ID, section_header, section_text, dialogue)
LLM     : openai/gpt-oss-120b via Groq (free tier)
Domain  : Medical
"""

import os
import sys
import pathlib
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

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

# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_mts_dialog(filepath: str, max_rows: int = None) -> list:
    """
    Loads MTS-Dialog CSV.
    Uses 'section_text' + 'dialogue' columns — one chunk per row.
    """
    df = pd.read_csv(filepath, encoding="utf-8")

    for col in ["section_text", "dialogue"]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Found: {list(df.columns)}"
            )

    if max_rows:
        df = df.head(max_rows)

    chunks = []
    for _, row in df.iterrows():
        section  = str(row["section_text"]).strip() if pd.notna(row["section_text"]) else ""
        dialogue = str(row["dialogue"]).strip()     if pd.notna(row["dialogue"])     else ""
        combined = " ".join(filter(None, [section, dialogue])).strip()
        if combined:
            chunks.append(combined)

    print(f"Loaded {len(chunks)} chunks from {filepath}")
    return chunks

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
            else:
                print(f"    [skip] unknown relation: '{rel}'")
    return triples

def process_chunk(text: str, kg: KnowledgeGraph) -> int:
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

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pass CSV path as argument or edit the default below
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/MTS-Dialog-TrainingSet.csv"

    if not pathlib.Path(csv_path).exists():
        print(f"ERROR: File not found → {csv_path}")
        print("Usage: python kg_construction.py data/MTS-Dialog-TrainingSet.csv")
        sys.exit(1)

    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print("Connected.\n")

    # Set max_rows=20 for a quick test, remove for full dataset
    chunks = load_mts_dialog(csv_path)

    total = 0
    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] {chunk[:80]}...")
        total += process_chunk(chunk, kg)

    stats = kg.get_stats()
    print(f"\n{'='*55}")
    print(f"DONE")
    print(f"  Triples inserted : {stats['triples']}")
    print(f"  Entities         : {stats['entities']}")
    print(f"  Relation types   : {stats['relation_types']}")
    print(f"{'='*55}")

    kg.close()