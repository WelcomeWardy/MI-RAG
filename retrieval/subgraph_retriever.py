import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

MAX_HOPS   = 2   # 1-hop and 2-hop paths
MAX_PATHS  = 50  # cap before MI scoring to avoid explosion


def _rel_to_snake(rel_type: str) -> str:
    return rel_type.lower().replace(" ", "_")


def retrieve_subgraph(entities: list[str]) -> list[list[tuple]]:
    """
    Input  : matched entity names from entity_matcher
    Output : list of paths, each path = list of (head, relation, tail) tuples

    Retrieves:
    - 1-hop: all direct triples for each seed entity
    - 2-hop: paths of length 2 connecting any two seed entities
    """
    if not entities:
        return []

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    paths  = []

    try:
        with driver.session() as session:

            # ── 1-hop: direct neighbours of each seed entity ──────────────────
            for entity in entities:
                result = session.run("""
                    MATCH (h:Entity {name: $name})-[r]->(t:Entity)
                    RETURN h.name AS head, type(r) AS rel, t.name AS tail
                    LIMIT 30
                """, name=entity)

                for record in result:
                    triple = (record["head"], _rel_to_snake(record["rel"]), record["tail"])
                    paths.append([triple])  # single-triple path

                # Also reverse direction
                result_rev = session.run("""
                    MATCH (h:Entity)-[r]->(t:Entity {name: $name})
                    RETURN h.name AS head, type(r) AS rel, t.name AS tail
                    LIMIT 30
                """, name=entity)

                for record in result_rev:
                    triple = (record["head"], _rel_to_snake(record["rel"]), record["tail"])
                    paths.append([triple])

            # ── 2-hop: paths between pairs of seed entities ───────────────────
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    result = session.run("""
                        MATCH p = (h:Entity {name: $e1})-[r1]->(m:Entity)-[r2]->(t:Entity {name: $e2})
                        RETURN h.name AS h, type(r1) AS r1, m.name AS mid,
                               type(r2) AS r2, t.name AS t
                        LIMIT 10
                    """, e1=entities[i], e2=entities[j])

                    for record in result:
                        path = [
                            (record["h"],   _rel_to_snake(record["r1"]), record["mid"]),
                            (record["mid"], _rel_to_snake(record["r2"]), record["t"]),
                        ]
                        paths.append(path)

    finally:
        driver.close()

    # Deduplicate paths
    seen = set()
    unique_paths = []
    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)

    # Cap before sending to MI scorer
    unique_paths = unique_paths[:MAX_PATHS]
    print(f"  [subgraph_retriever] {len(unique_paths)} unique paths retrieved")
    return unique_paths


if __name__ == "__main__":
    from keyword_extractor import extract_keywords
    from entity_matcher import match_entities

    kws      = extract_keywords("What are the symptoms and treatments for diabetes?")
    entities = match_entities(kws)
    paths    = retrieve_subgraph(entities)

    print(f"\nSample paths:")
    for p in paths[:5]:
        print(f"  {p}")