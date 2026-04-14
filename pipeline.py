import os
from dotenv import load_dotenv

from retrieval.keyword_extractor import extract_keywords
from retrieval.entity_matcher import match_entities
from retrieval.subgraph_retriever import retrieve_subgraph
from scoring.mi_scorer import score_paths
from scoring.pruner import prune
from aggregation.aggregator import aggregate
from answering.mapreduce_chain import mapreduce_answer

load_dotenv()

def run_pipeline(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Step 1 — Extract keywords
    print("\n[Step 1] Keyword Extraction")
    keywords = extract_keywords(question)

    # Step 2 — Match entities in KG
    print("\n[Step 2] Entity Matching")
    entities = match_entities(keywords)
    if not entities:
        return "No relevant entities found in the knowledge graph."

    # Step 3 — Retrieve subgraph paths
    print("\n[Step 3] Subgraph Retrieval")
    paths = retrieve_subgraph(entities)
    if not paths:
        return "No relevant paths found in the knowledge graph."

    # Step 4 — MI scoring + reranking
    print("\n[Step 4] MI Scoring")
    ranked_paths = score_paths(paths, question)

    # Step 5 — Pruning
    print("\n[Step 5] Pruning")
    pruned = prune(ranked_paths, question)

    # Step 6 — Aggregation
    print("\n[Step 6] Aggregation")
    sentences = aggregate(pruned)

    # Step 7 — MapReduce answering
    print("\n[Step 7] MapReduce Answering")
    answer = mapreduce_answer(sentences, question)

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{answer}")
    print(f"{'='*60}")
    return answer


if __name__ == "__main__":
    questions = [
        "What are the symptoms and treatments for diabetes?",
        "What medications are used for hypertension?",
        "What diagnostic tests are used for cancer?",
    ]
    for q in questions:
        run_pipeline(q)
        print("\n")