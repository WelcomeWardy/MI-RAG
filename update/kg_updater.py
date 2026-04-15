import os
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import lru_cache

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

SIMILARITY_THRESHOLD = 0.85  # if new triple is this similar to existing → skip

ALLOWED_RELATIONS = [
    "can_treat_disease", "may_suffer_from_disease", "should_eat",
    "common_medication_is", "accompanied_by_symptoms_of",
    "should_avoid_eating", "symptoms_are", "diagnostic_tests",
]

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Paper Section 3.5: head nodes come from retrieval (cosine similarity step 3.2)
# LLM is prompted with head node + allowed relations + answer to build new triples
UPDATE_PROMPT = PromptTemplate.from_template("""
You are a medical knowledge graph expert.
Given the answer below, extract new knowledge triples using ONLY the head entity provided.

Head entity (must be used as head): {head_entity}

Allowed relations (use exactly as written):
{allowed_relations}

Answer: {answer}

Format — one triple per line:
{head_entity} | relation | tail_entity

Rules:
- Head entity must be exactly: {head_entity}
- Only use relations from the allowed list
- Tail: concise lowercase noun phrase
- Output ONLY triples, no explanations

Triples:
""")

update_chain = UPDATE_PROMPT | llm | StrOutputParser()


@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def _triple_to_text(h: str, r: str, t: str) -> str:
    return f"{h} {r.replace('_', ' ')} {t}"


def _get_one_hop_neighbours(driver, head: str) -> list[tuple]:
    """Get existing 1-hop triples from head node in KG."""
    with driver.session() as session:
        result = session.run("""
            MATCH (h:Entity {name: $name})-[r]->(t:Entity)
            RETURN h.name AS head, type(r) AS rel, t.name AS tail
        """, name=head)
        return [(r["head"], r["rel"].lower(), r["tail"]) for r in result]


def _is_similar_to_existing(
    new_triple: tuple, existing_triples: list[tuple], threshold: float
) -> bool:
    """
    Paper Section 3.5: check cosine similarity of new triple against
    existing 1-hop neighbours. If highly similar → skip (don't insert).
    """
    if not existing_triples:
        return False

    embedder = _get_embedder()
    new_text      = _triple_to_text(*new_triple)
    existing_texts = [_triple_to_text(*t) for t in existing_triples]

    new_vec      = embedder.encode([new_text],      normalize_embeddings=True)[0]
    existing_vecs = embedder.encode(existing_texts, normalize_embeddings=True)

    sims = np.dot(existing_vecs, new_vec)
    max_sim = float(np.max(sims))

    if max_sim >= threshold:
        print(f"    [kg_updater] SKIP (sim={max_sim:.3f} ≥ {threshold}): {new_text}")
        return True
    return False


def _parse_triples(raw: str, expected_head: str) -> list[tuple]:
    triples = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            head, rel, tail = parts
            rel_clean = rel.lower().replace(" ", "_").replace("-", "_")
            # Enforce head node from retrieval (paper requirement)
            if rel_clean in ALLOWED_RELATIONS:
                triples.append((expected_head.lower(), rel_clean, tail.lower()))
    return triples


def _insert_triple(driver, head: str, relation: str, tail: str):
    rel_label = relation.upper()
    query = f"""
    MERGE (h:Entity {{name: $head}})
    MERGE (t:Entity {{name: $tail}})
    MERGE (h)-[r:{rel_label}]->(t)
    """
    with driver.session() as session:
        session.run(query, head=head, tail=tail)


def update_kg(answer: str, matched_entities: list[str]) -> int:
    """
    Paper Section 3.5 implementation:
    - matched_entities: head nodes from cosine similarity step (entity_matcher output)
    - For each head node, prompt LLM to generate new triples using answer
    - Check each new triple against existing 1-hop neighbours via cosine similarity
    - Only insert if not highly similar to existing triples

    Input  : final answer string, list of matched entity names from retrieval
    Output : number of new triples inserted
    """
    if not matched_entities:
        print("  [kg_updater] No head entities provided.")
        return 0

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    total_inserted = 0

    try:
        for head_entity in matched_entities:
            print(f"\n  [kg_updater] Processing head entity: '{head_entity}'")

            # Generate new triples with this head node
            raw = update_chain.invoke({
                "head_entity": head_entity,
                "answer": answer,
                "allowed_relations": "\n".join(f"- {r}" for r in ALLOWED_RELATIONS),
            })

            new_triples = _parse_triples(raw, head_entity)
            if not new_triples:
                print(f"    [kg_updater] No triples generated for '{head_entity}'")
                continue

            # Get existing 1-hop neighbours for similarity check
            existing = _get_one_hop_neighbours(driver, head_entity)
            print(f"    [kg_updater] {len(existing)} existing 1-hop neighbours")

            # Insert only non-similar triples (paper Section 3.5)
            for triple in new_triples:
                if not _is_similar_to_existing(triple, existing, SIMILARITY_THRESHOLD):
                    _insert_triple(driver, *triple)
                    print(f"    [kg_updater] + ({triple[0]}) -[{triple[1]}]-> ({triple[2]})")
                    total_inserted += 1
                    # Update existing list so subsequent triples in same batch
                    # are also checked against newly inserted ones
                    existing.append(triple)

    finally:
        driver.close()

    print(f"  [kg_updater] {total_inserted} new triples inserted.")
    return total_inserted


if __name__ == "__main__":
    fake_answer = (
        "Diabetes symptoms include fatigue, frequent urination, and blurred vision. "
        "Insulin is the primary medication. Blood glucose tests are used for diagnosis."
    )
    fake_entities = ["diabetes", "insulin"]
    update_kg(fake_answer, fake_entities)