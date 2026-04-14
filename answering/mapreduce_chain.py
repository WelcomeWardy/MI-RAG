import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Map step 
map_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Reduce step 
reduce_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)

MAP_PROMPT = PromptTemplate.from_template("""
You are a medical expert. Using ONLY the fact below, give a partial answer to the question.
1-2 sentences only.

Fact: {sentence}
Question: {question}

Partial answer:
""")

REDUCE_PROMPT = PromptTemplate.from_template("""
You are a medical expert. Combine these partial answers into one final, 
coherent, complete answer to the question.

Question: {question}

Partial answers:
{partial_answers}

Final answer:
""")

map_chain    = MAP_PROMPT    | map_llm    | StrOutputParser()
reduce_chain = REDUCE_PROMPT | reduce_llm | StrOutputParser()


def mapreduce_answer(sentences: list[str], question: str) -> str:
    """
    Input  : sentences D from aggregator, question Q
    Output : final answer string
    """

    # MAP — each sentence → partial answer
    print(f"  Map step: {len(sentences)} sentences...")
    partial_answers = []
    for i, sentence in enumerate(sentences):
        try:
            partial = map_chain.invoke({
                "sentence": sentence,
                "question": question,
            })
            partial_answers.append(partial.strip())
            print(f"    [{i+1}] {partial.strip()[:80]}...")
        except Exception as e:
            print(f"    [{i+1}] ERROR: {e}")
            continue

    if not partial_answers:
        return "Could not generate answer."

    # REDUCE — combine all partials → final answer
    print(f"\n  Reduce step...")
    combined = "\n".join(f"{i+1}. {p}" for i, p in enumerate(partial_answers))

    final = reduce_chain.invoke({
        "question": question,
        "partial_answers": combined,
    })

    print(f"\n  Final answer: {final.strip()}")
    return final.strip()


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulating aggregator output
    FAKE_SENTENCES = [
        "Individuals with diabetes often experience symptoms such as fatigue.",
        "The most common medication used to manage diabetes is insulin.",
        "A blood glucose test is a diagnostic test used to determine if a patient has diabetes.",
    ]

    FAKE_QUESTION = "What are the symptoms and treatments for diabetes?"

    answer = mapreduce_answer(FAKE_SENTENCES, FAKE_QUESTION)
    print("\n" + "="*50)
    print("FINAL ANSWER:")
    print(answer)