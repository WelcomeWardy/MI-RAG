import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2-1.5B"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"  [mi_scorer] Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        _model.eval()
        print(f"  [mi_scorer] Model loaded.")


def _token_log_probs(input_ids: torch.Tensor) -> torch.Tensor:
    """Return per-token log probs for the full sequence."""
    with torch.no_grad():
        logits = _model(input_ids).logits  # (1, seq, vocab)
    shift_logits = logits[0, :-1, :]       # predicts token[i+1]
    shift_labels = input_ids[0, 1:]
    lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    return lp[range(len(shift_labels)), shift_labels]  # (seq-1,)


def _score_path_tokens_given_context(
    context_text: str, path_text: str
) -> float:
    """
    Computes sum of log P(path_token_i | context, path_tokens_<i).
    This estimates log p(path | context) via autoregressive decomposition.
    Used for both numerator (context = Q + e1) and denominator (context = e1).
    Implements Eq. 5 and 6 from the paper.
    """
    _load_model()

    context_ids = _tokenizer(context_text, return_tensors="pt")["input_ids"]
    full_ids     = _tokenizer(context_text + " " + path_text, return_tensors="pt")["input_ids"]

    ctx_len = context_ids.shape[1]
    all_lp  = _token_log_probs(full_ids)

    # Only sum log probs over path tokens (after context)
    path_lp = all_lp[ctx_len - 1:]
    return float(path_lp.sum())


def _path_to_parts(path: list[tuple]) -> tuple[str, str]:
    """
    Splits a path into:
      - seed entity (e1): head of first triple
      - path body (r1, e2, r2, e3 ...): rest of the path tokens
    As per the paper's notation.
    """
    if not path:
        return "", ""

    e1 = path[0][0]
    # Path body = relations and tail entities
    body_parts = []
    for head, rel, tail in path:
        body_parts.append(rel.replace("_", " "))
        body_parts.append(tail)
    return e1, " ".join(body_parts)


def _mi_score(path: list[tuple], question: str) -> float:
    """
    Implements Eq. 3 from the paper:

        score = log [ p(path_body | Q, e1) / p(path_body | e1) ]
              = log p(path_body | Q, e1) - log p(path_body | e1)

    Numerator   : p(r1, e2, r2, e3, ... | Q, e1)  — Eq. 5
    Denominator : p(r1, e2, r2, e3, ... | e1)      — Eq. 6

    Higher score = path is more informative given the question
    than it would be without it.
    """
    e1, path_body = _path_to_parts(path)
    if not path_body:
        return float("-inf")

    # Numerator: condition on Q AND e1
    numerator = _score_path_tokens_given_context(
        context_text=f"{question} {e1}",
        path_text=path_body,
    )
    # Denominator: condition on e1 only
    denominator = _score_path_tokens_given_context(
        context_text=e1,
        path_text=path_body,
    )

    return numerator - denominator


def score_paths(paths: list[list[tuple]], question: str) -> list[list[tuple]]:
    """
    Input  : paths from subgraph_retriever, question string
    Output : same paths sorted descending by MI score (Eq. 3–7)
    """
    if not paths:
        return []

    print(f"  [mi_scorer] Scoring {len(paths)} paths...")
    scored = []

    for i, path in enumerate(paths):
        mi = _mi_score(path, question)
        e1, body = _path_to_parts(path)
        print(f"    [{i+1}/{len(paths)}] MI={mi:.4f} | e1={e1} | body={body[:50]}...")
        scored.append((mi, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [path for _, path in scored]
    e1_top, body_top = _path_to_parts(ranked[0])
    print(f"  [mi_scorer] Top path MI: e1={e1_top} | {body_top[:60]}")
    return ranked


if __name__ == "__main__":
    FAKE_PATHS = [
        [("diabetes", "symptoms_are", "fatigue"),
         ("diabetes", "symptoms_are", "frequent urination")],
        [("diabetes", "common_medication_is", "insulin")],
        [("diabetes", "diagnostic_tests", "blood glucose test")],
    ]
    Q = "What are the symptoms and treatments for diabetes?"
    ranked = score_paths(FAKE_PATHS, Q)
    print("\nRanked paths:")
    for p in ranked:
        print(f"  {p}")