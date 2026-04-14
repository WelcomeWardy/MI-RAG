import os
import re
from functools import lru_cache
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

SIMILARITY_THRESHOLD = 0.75  # tune this if too many/few matches


@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")  # same model as pruner.py


def _get_all_entity_names(driver) -> list[str]:
    with driver.session() as session:
        result = session.run("MATCH (e:Entity) RETURN e.name AS name")
        return [r["name"] for r in result]


def match_entities(keywords: list[str]) -> list[str]:
    """
    Input  : list of keyword strings from keyword_extractor
    Output : list of entity names that exist in the KG (matched or exact)

    Strategy:
    1. Exact match first (fast)
    2. Semantic similarity via sentence-transformers for misses
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        all_entities = _get_all_entity_names(driver)
    finally:
        driver.close()

    if not all_entities:
        print("  [entity_matcher] No entities in KG yet.")
        return []

    entity_set = set(all_entities)
    matched = []
    unmatched = []

    # Pass 1: exact match
    for kw in keywords:
        kw_lower = kw.strip().lower()
        if kw_lower in entity_set:
            matched.append(kw_lower)
        else:
            unmatched.append(kw_lower)

    # Pass 2: semantic similarity for unmatched keywords
    if unmatched:
        embedder = _get_embedder()
        entity_vecs = embedder.encode(all_entities, normalize_embeddings=True)
        kw_vecs     = embedder.encode(unmatched,    normalize_embeddings=True)

        for i, kw in enumerate(unmatched):
            sims = np.dot(entity_vecs, kw_vecs[i])
            best_idx  = int(np.argmax(sims))
            best_score = float(sims[best_idx])

            if best_score >= SIMILARITY_THRESHOLD:
                matched.append(all_entities[best_idx])
                print(f"  [entity_matcher] '{kw}' → '{all_entities[best_idx]}' (sim={best_score:.3f})")
            else:
                print(f"  [entity_matcher] '{kw}' → no match (best={best_score:.3f})")

    # Deduplicate
    seen = set()
    result = []
    for e in matched:
        if e not in seen:
            seen.add(e)
            result.append(e)

    print(f"  [entity_matcher] matched entities: {result}")
    return result


if __name__ == "__main__":
    from keyword_extractor import extract_keywords
    kws = extract_keywords("What are the symptoms and treatments for diabetes?")
    print(match_entities(kws))