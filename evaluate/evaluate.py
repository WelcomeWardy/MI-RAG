import os, json, time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,
               api_key=os.getenv("GROQ_API_KEY"))

EVAL_PROMPT = PromptTemplate.from_template("""
You are evaluating a medical QA system.

Question        : {question}
Generated answer: {generated}

Score on 3 criteria (each 0.0 to 1.0):
1. Faithfulness  : Is the answer factually plausible for the question?
2. Relevance     : Does it directly address the question?
3. Completeness  : Is the answer sufficiently detailed?

Output JSON only:
{{"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0}}
""")

eval_chain = EVAL_PROMPT | llm | StrOutputParser()


def get_questions_from_kg(dataset_tag: str, limit: int = 10) -> list[str]:
    """Pull questions stored during KG construction for a specific dataset."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (q:Question {dataset: $tag})
                RETURN q.text AS text
                LIMIT $limit
            """, tag=dataset_tag, limit=limit)
            questions = [r["text"] for r in result]
    finally:
        driver.close()

    print(f"  [eval] Loaded {len(questions)} questions for dataset='{dataset_tag}'")
    return questions


def evaluate_single(question: str, generated: str) -> dict:
    try:
        raw = eval_chain.invoke({"question": question, "generated": generated})
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        scores["overall"] = round(
            (scores["faithfulness"] + scores["relevance"] + scores["completeness"]) / 3, 4
        )
        return scores
    except Exception as e:
        print(f"  [eval] Error: {e}")
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "overall": 0}


def evaluate_dataset(dataset_tag: str, pipeline_fn, limit: int = 10) -> dict:
    """
    Pulls questions from Neo4j by dataset tag, runs pipeline, scores answers.
    """
    questions = get_questions_from_kg(dataset_tag, limit=limit)
    if not questions:
        print(f"  [eval] No questions found for tag='{dataset_tag}'. Run construction.py first.")
        return {}

    all_scores = []
    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {question[:70]}...")
        try:
            generated = pipeline_fn(question)
        except Exception as e:
            print(f"  [pipeline error] {e}")
            generated = ""

        scores = evaluate_single(question, generated)
        scores["question"] = question
        all_scores.append(scores)
        print(f"  F={scores['faithfulness']} R={scores['relevance']} C={scores['completeness']} Overall={scores['overall']}")
        time.sleep(3)  # rate limit guard

    n = len(all_scores)
    agg = {
        "dataset":       dataset_tag,
        "n":             n,
        "faithfulness":  round(sum(s["faithfulness"]  for s in all_scores) / n, 4),
        "relevance":     round(sum(s["relevance"]     for s in all_scores) / n, 4),
        "completeness":  round(sum(s["completeness"]  for s in all_scores) / n, 4),
        "overall":       round(sum(s["overall"]       for s in all_scores) / n, 4),
    }

    print(f"\n{'='*50}")
    print(f"RESULTS — {dataset_tag} (n={n})")
    print(f"  Faithfulness : {agg['faithfulness']}")
    print(f"  Relevance    : {agg['relevance']}")
    print(f"  Completeness : {agg['completeness']}")
    print(f"  Overall      : {agg['overall']}")
    print(f"{'='*50}")
    return agg


if __name__ == "__main__":
    from pipeline import run_pipeline

    # Evaluate separately per dataset — shows results for each DB
    for tag in ["MTS-Dialog", "MedQuAD"]:
        evaluate_dataset(tag, run_pipeline, limit=5)