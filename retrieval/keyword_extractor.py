import os
import re
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

KEYWORD_PROMPT = PromptTemplate.from_template("""
You are a medical NLP expert. Extract medical keywords from the question below.
Focus on: diseases, symptoms, medications, diagnostic tests, body parts, treatments.

Rules:
- Output a comma-separated list only
- Lowercase only
- No explanations, no numbering
- Max 10 keywords

Question: {question}

Keywords:
""")

keyword_chain = KEYWORD_PROMPT | llm | StrOutputParser()


def extract_keywords(question: str) -> list[str]:
    """
    Input  : natural language question string
    Output : list of lowercase keyword strings
    """
    raw = keyword_chain.invoke({"question": question})
    keywords = [
        k.strip().lower()
        for k in raw.split(",")
        if k.strip() and len(k.strip()) > 2
    ]
    # Also add simple noun chunks via regex as fallback
    fallback = [
        t.lower() for t in re.findall(r"\b[a-zA-Z]{4,}\b", question)
        if t.lower() not in {"what", "does", "this", "that", "with", "from", "have", "your", "when"}
    ]
    # Merge, deduplicate, preserve order
    seen = set()
    merged = []
    for k in keywords + fallback:
        if k not in seen:
            seen.add(k)
            merged.append(k)

    print(f"  [keyword_extractor] {merged}")
    return merged


if __name__ == "__main__":
    q = "What are the symptoms and treatments for diabetes?"
    print(extract_keywords(q))