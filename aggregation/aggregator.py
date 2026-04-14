import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

AGGREGATION_PROMPT = PromptTemplate.from_template("""
You are a medical expert. Convert these knowledge graph triples into clear, 
fluent medical sentences. One sentence per triple. No bullet points.

Triples:
{triples}

Sentences:
""")

aggregation_chain = AGGREGATION_PROMPT | llm | StrOutputParser()

def aggregate(pruned_triples: list[tuple]) -> list[str]:
    """
    Input  : pruned triples Gp from pruner
    Output : list of natural language sentences D
    """
    formatted = "\n".join(
        f"- {h} {r.replace('_', ' ')} {t}" 
        for h, r, t in pruned_triples
    )

    raw = aggregation_chain.invoke({"triples": formatted})

    # Split into sentences
    sentences = [s.strip() for s in raw.strip().split("\n") if s.strip()]
    print(f"  Aggregator: {len(sentences)} sentences generated")
    for s in sentences:
        print(f"    → {s}")

    return sentences


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulating pruner output
    FAKE_GP = [
        ("diabetes", "symptoms_are", "fatigue"),
        ("diabetes", "common_medication_is", "insulin"),
        ("diabetes", "diagnostic_tests", "blood glucose test"),
    ]

    result = aggregate(FAKE_GP)
    print("\nAggregated sentences D:")
    for s in result:
        print(f"  {s}")