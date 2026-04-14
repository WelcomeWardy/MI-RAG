import os
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Qwen2-1.5B local — same as discussed
MODEL_NAME = "Qwen/Qwen2-1.5B"

_tokenizer = None
_model     = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        print(f"  [mi_scorer] Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,   # float32 for CPU safety
            device_map="cpu",
        )
        _model.eval()
        print(f"  [mi_scorer] Model loaded.")


def _log_prob(context: str, continuation: str) -> float:
    """
    Compute log P(continuation | context) using Qwen2 log-probs.
    This is the core MI approximation:
        MI(path; question) ≈ log P(question | path) - log P(question)
    We compute log P(question_tokens | path_context) via teacher-forcing.
    """
    _load_model()

    prompt   = context + " " + continuation
    inputs   = _tokenizer(prompt,   return_tensors="pt")
    ctx_ids  = _tokenizer(context,  return_tensors="pt")["input_ids"]

    ctx_len  = ctx_ids.shape[1]
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = _model(input_ids, labels=input_ids)
        # outputs.loss = mean NLL over ALL tokens
        # We want NLL only over the continuation tokens
        logits = outputs.logits  # (1, seq_len, vocab)

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[0, :-1, :]          # (seq_len-1, vocab)
    shift_labels = input_ids[0, 1:]           # (seq_len-1,)

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

    # Only sum over continuation portion
    continuation_log_probs = token_log_probs[ctx_len - 1:]
    return float(continuation_log_probs.sum())


def _path_to_text(path: list[tuple]) -> str:
    return " ".join(
        f"{h} {r.replace('_', ' ')} {t}"
        for h, r, t in path
    )


def score_paths(paths: list[list[tuple]], question: str) -> list[list[tuple]]:
    """
    Input  : paths from subgraph_retriever, question string
    Output : same paths, sorted descending by MI score

    MI score per path ≈ log P(question | path_text) - log P(question | "")
    Higher = path is more informative about the question.
    """
    if not paths:
        return []

    print(f"  [mi_scorer] Scoring {len(paths)} paths...")

    # Baseline: log P(question) with empty context
    baseline = _log_prob("", question)

    scored = []
    for i, path in enumerate(paths):
        path_text = _path_to_text(path)
        lp        = _log_prob(path_text, question)
        mi        = lp - baseline
        scored.append((mi, path))
        print(f"    [{i+1}/{len(paths)}] MI={mi:.4f} | {path_text[:60]}...")

    # Sort descending — highest MI first
    scored.sort(key=lambda x: x[0], reverse=True)

    ranked_paths = [path for _, path in scored]
    print(f"  [mi_scorer] Top path: {_path_to_text(ranked_paths[0])[:80]}")
    return ranked_paths


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