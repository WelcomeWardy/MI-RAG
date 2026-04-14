import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

ALLOWED_RELATIONS = [
    "can_treat_disease", "may_suffer_from_disease", "should_eat",
    "common_medication_is", "accompanied_by_symptoms_of",
    "should_avoid_eating", "symptoms_are", "diagnostic_tests",
]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

UPDATE_PROMPT = PromptTemplate.from_template("""
You are a medical knowledge graph expert.
Given the question and the answer below, extract any NEW knowledge triples
that are not already implied by the question alone.

Format — one triple per line:
head_entity | relation | tail_entity

Allowed relations only:
{allowed_relations}

Rules:
- Entities: concise lowercase English noun phrases
- Only use relations from the allowed list
- Output ONLY triples, no explanations

Question: {question}
Answer: {answer}

Triples:
""")

update_chain = UPDATE_PROMPT | llm | StrOutputParser()


def _parse_triples(raw: str) -> list[tuple]:
    triples = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            head, rel, tail = parts
            rel_clean = rel.lower().replace(" ", "_").replace("-", "_")
            if rel_clean in ALLOWED_RELATIONS:
                triples.append((head.lower(), rel_clean, tail.lower()))
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


def update_kg(question: str, answer: str) -> int:
    """
    Input  : question and final answer from pipeline
    Output : number of new triples inserted

    Extracts new triples from the Q&A and inserts them into Neo4j.
    This closes the feedback loop — KG improves over time.
    """
    raw = update_chain.invoke({
        "question": question,
        "answer": answer,
        "allowed_relations": "\n".join(f"- {r}" for r in ALLOWED_RELATIONS),
    })

    triples = _parse_triples(raw)
    if not triples:
        print("  [kg_updater] No new triples extracted.")
        return 0

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        for head, rel, tail in triples:
            _insert_triple(driver, head, rel, tail)
            print(f"  [kg_updater] + ({head}) -[{rel}]-> ({tail})")
    finally:
        driver.close()

    print(f"  [kg_updater] {len(triples)} new triples inserted.")
    return len(triples)


if __name__ == "__main__":
    Q = "What are the symptoms of diabetes?"
    A = "Diabetes symptoms include fatigue, frequent urination, and blurred vision. Insulin is the common medication."
    update_kg(Q, A)