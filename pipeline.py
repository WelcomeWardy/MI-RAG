import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from retrieval.keyword_extractor import extract_keywords
from retrieval.entity_matcher import match_entities
from retrieval.subgraph_retriever import retrieve_subgraph
from scoring.mi_scorer import score_paths
from scoring.pruner import prune
from aggregation.aggregator import aggregate
from answering.mapreduce_chain import mapreduce_answer
from update.kg_updater import update_kg

load_dotenv()


def _ddg_fallback(question: str) -> str:
    """
    Paper Section 3.4: when Gp is empty (question outside KG + LLM scope),
    fall back to DuckDuckGo search.
    """
    print("  [pipeline] Gp empty — falling back to DuckDuckGo...")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(question, max_results=5)
            snippets = [r.get("body", "") for r in results if r.get("body")]
            if snippets:
                combined = " ".join(snippets[:3])
                print(f"  [pipeline] DDG result: {combined[:100]}...")
                return combined
    except Exception as e:
        print(f"  [pipeline] DDG error: {e}")
    return "Could not find an answer."


def run_pipeline(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    # Step 1 — Keywords
    print("\n[Step 1] Keyword Extraction")
    keywords = extract_keywords(question)

    # Step 2 — Entity matching (cosine similarity → entity set E)
    print("\n[Step 2] Entity Matching")
    entities = match_entities(keywords)
    if not entities:
        print("  No entities matched — DDG fallback")
        return _ddg_fallback(question)

    # Step 3 — Subgraph retrieval
    print("\n[Step 3] Subgraph Retrieval")
    paths = retrieve_subgraph(entities)
    if not paths:
        print("  No paths found — DDG fallback")
        return _ddg_fallback(question)

    # Step 4 — MI scoring (Eq. 3–7)
    print("\n[Step 4] MI Scoring")
    ranked_paths = score_paths(paths, question)

    # Step 5 — Self-pruning (Eq. 8–9)
    print("\n[Step 5] Pruning")
    pruned = prune(ranked_paths, question)

    # Step 6 — Check if Gp is empty → DDG fallback (Section 3.4)
    if not pruned:
        print("  Pruned subgraph empty — DDG fallback")
        return _ddg_fallback(question)

    # Step 7 — Aggregation: triples → natural language sentences D
    print("\n[Step 6] Aggregation")
    sentences = aggregate(pruned)

    # Step 8 — MapReduce answering: Map(D) → Reduce → final answer
    print("\n[Step 7] MapReduce Answering")
    answer = mapreduce_answer(sentences, question)

    # Step 9 — KG update using matched entities as head nodes (Section 3.5)
    print("\n[Step 8] KG Update")
    update_kg(answer, entities)

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