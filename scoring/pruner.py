import os
import itertools
import re
import difflib
from functools import lru_cache

import numpy as np
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

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

PRUNE_PROMPT = PromptTemplate.from_template("""
You are a medical expert. Given these facts, answer the question as confidently as possible.
Be concise — 1-2 sentences only.

Facts:
{triples}

Question: {question}

Answer:
""")

prune_chain = PRUNE_PROMPT | llm | StrOutputParser()

# ── Proxy instability via repeated sampling ───
@lru_cache(maxsize=1)
def _get_embedder():
    if SentenceTransformer is None:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


def _semantic_similarity(text_a: str, text_b: str) -> float:
    embedder = _get_embedder()
    if embedder is None:
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()

    vecs = embedder.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", text.lower())).strip()


def estimate_instability(triples_str: str, question: str, n_samples: int = 3) -> float:
    """
    Sample the LLM n times for same input.
    Instability proxy = mean pairwise semantic dissimilarity.
    High variance = unstable = uncertain = weak combo.
    Low variance  = stable   = confident = strong combo.
    """
    responses = []
    for _ in range(n_samples):
        try:
            r = prune_chain.invoke({"triples": triples_str, "question": question})
            responses.append(_normalize_text(r))
        except Exception as e:
            print(f"    [sample error] {e}")
            return float("inf")

    embedder = _get_embedder()
    if embedder is None:
        # Fallback when sentence-transformers is unavailable.
        scores = []
        for a, b in itertools.combinations(responses, 2):
            similarity = difflib.SequenceMatcher(None, a, b).ratio()
            scores.append(1 - similarity)
        return sum(scores) / len(scores) if scores else float("inf")

    embeddings = embedder.encode(responses, normalize_embeddings=True)

    scores = []
    for left, right in itertools.combinations(range(len(embeddings)), 2):
        similarity = float(np.dot(embeddings[left], embeddings[right]))
        scores.append(1 - similarity)

    return sum(scores) / len(scores) if scores else float("inf")

# ── Triple formatting ────────

def format_triples(path: list[tuple]) -> str:
    return "\n".join(f"- {h} {r.replace('_', ' ')} {t}" for h, r, t in path)


def _question_terms(question: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 3
    }


def _coverage_score(combo: list[tuple], question: str) -> float:
    terms = _question_terms(question)
    if not terms:
        return 0.0

    combo_text = " ".join(str(part).lower() for triple in combo for part in triple)
    combo_terms = set(re.findall(r"[a-z0-9]+", combo_text))
    return len(terms & combo_terms) / len(terms)


def _semantic_relevance(combo: list[tuple], question: str) -> float:
    combo_text = " ".join(f"{h} {r.replace('_', ' ')} {t}" for h, r, t in combo)
    return _semantic_similarity(_normalize_text(combo_text), _normalize_text(question))

# ── Pruner ───────

def prune(
    paths: list[list[tuple]],
    question: str,
    max_triples: int = 6,
    alpha_instability: float = 0.6,
    beta_relevance: float = 0.3,
    gamma_coverage: float = 0.1,
) -> list[tuple]:
    """
    Flattens paths → unique triples → enumerates combos
    MIRAG-style heuristic:
    choose triples that are semantically relevant to the query and produce
    stable sampled answers.
    Returns pruned subgraph Gp.
    """
    seen = set()
    all_triples = []
    for path in paths:
        for triple in path:
            if triple not in seen:
                seen.add(triple)
                all_triples.append(triple)

    all_triples = all_triples[:max_triples]
    print(f"  Pruner: {len(all_triples)} unique triples, enumerating combos...")

    best_combo   = all_triples
    best_score = float("inf")

    for size in range(2, len(all_triples) + 1):
        for combo in itertools.combinations(all_triples, size):
            formatted = format_triples(list(combo))
            instability = estimate_instability(formatted, question, n_samples=3)
            relevance = _semantic_relevance(list(combo), question)
            coverage = _coverage_score(list(combo), question)
            score = (
                (alpha_instability * instability)
                - (beta_relevance * relevance)
                - (gamma_coverage * coverage)
            )
            print(
                f"    size={size} | instability={instability:.4f} | relevance={relevance:.4f} | coverage={coverage:.2f} | score={score:.4f} | {[c[0]+'-'+c[2] for c in combo]}"
            )

            if score < best_score:
                best_score = score
                best_combo   = list(combo)

    print(f"\n  Best combo ({len(best_combo)} triples) | score={best_score:.4f}")
    return best_combo


# ── Test ───────────────

if __name__ == "__main__":
    FAKE_PATHS = [
        [
            ("diabetes", "symptoms_are", "fatigue"),
            ("diabetes", "symptoms_are", "frequent urination"),
        ],
        [
            ("diabetes", "common_medication_is", "insulin"),
            ("insulin", "can_treat_disease", "diabetes"),
        ],
        [
            ("diabetes", "diagnostic_tests", "blood glucose test"),
            ("blood glucose test", "accompanied_by_symptoms_of", "hyperglycemia"),
        ],
    ]

    FAKE_QUESTION = "What are the symptoms and treatments for diabetes?"

    result = prune(FAKE_PATHS, FAKE_QUESTION)
    print("\nPruned subgraph Gp:")
    for triple in result:
        print(f"  {triple}")