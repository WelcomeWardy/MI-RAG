import os
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

EVAL_PROMPT = PromptTemplate.from_template("""
You are evaluating a medical QA system. Score the generated answer vs the reference answer.

Question       : {question}
Reference answer: {reference}
Generated answer: {generated}

Score on these 3 criteria (each 0.0 to 1.0):
1. Faithfulness  : Is the generated answer factually consistent with the reference?
2. Relevance     : Does it directly answer the question?
3. Completeness  : Does it cover the key points from the reference?

Output JSON only, no explanation:
{{"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0}}
""")

eval_chain = EVAL_PROMPT | llm | StrOutputParser()


def evaluate_single(question: str, reference: str, generated: str) -> dict:
    raw = eval_chain.invoke({
        "question":  question,
        "reference": reference,
        "generated": generated,
    })
    try:
        # Strip markdown fences if present
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
        scores["overall"] = round(
            (scores["faithfulness"] + scores["relevance"] + scores["completeness"]) / 3, 4
        )
        return scores
    except Exception as e:
        print(f"  [eval] JSON parse error: {e} | raw: {raw}")
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "overall": 0}


def evaluate_dataset(qa_pairs: list[dict], pipeline_fn) -> dict:
    """
    Input  : list of {"question": ..., "answer": ...} dicts (ground truth)
             pipeline_fn — callable that takes question → generated answer string
    Output : aggregate scores dict

    Example qa_pairs entry:
        {"question": "What are symptoms of diabetes?", "answer": "Fatigue, frequent urination..."}
    """
    all_scores = []

    for i, item in enumerate(qa_pairs):
        question  = item["question"]
        reference = item["answer"]

        print(f"\n[{i+1}/{len(qa_pairs)}] {question[:60]}...")

        try:
            generated = pipeline_fn(question)
        except Exception as e:
            print(f"  [eval] Pipeline error: {e}")
            generated = ""

        scores = evaluate_single(question, reference, generated)
        scores["question"] = question
        all_scores.append(scores)

        print(f"  Faithfulness={scores['faithfulness']} | "
              f"Relevance={scores['relevance']} | "
              f"Completeness={scores['completeness']} | "
              f"Overall={scores['overall']}")

        # Rate limit guard
        time.sleep(3)

    # Aggregate
    n = len(all_scores)
    agg = {
        "faithfulness":  round(sum(s["faithfulness"]  for s in all_scores) / n, 4),
        "relevance":     round(sum(s["relevance"]     for s in all_scores) / n, 4),
        "completeness":  round(sum(s["completeness"]  for s in all_scores) / n, 4),
        "overall":       round(sum(s["overall"]       for s in all_scores) / n, 4),
        "n":             n,
    }

    print(f"\n{'='*50}")
    print(f"EVAL RESULTS (n={n})")
    print(f"  Faithfulness : {agg['faithfulness']}")
    print(f"  Relevance    : {agg['relevance']}")
    print(f"  Completeness : {agg['completeness']}")
    print(f"  Overall      : {agg['overall']}")
    print(f"{'='*50}")

    return agg


if __name__ == "__main__":
    # Quick smoke test with fake data
    from pipeline import run_pipeline

    TEST_QA = [
        {
            "question": "What are the symptoms and treatments for diabetes?",
            "answer": "Diabetes symptoms include fatigue, frequent urination, blurred vision. "
                      "Treatment includes insulin therapy and blood glucose monitoring."
        },
        {
            "question": "What medications are used for hypertension?",
            "answer": "Common medications for hypertension include ACE inhibitors, beta blockers, "
                      "and calcium channel blockers."
        },
    ]

    results = evaluate_dataset(TEST_QA, run_pipeline)
    print(json.dumps(results, indent=2))