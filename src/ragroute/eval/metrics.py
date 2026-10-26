"""Primary quality metrics (pre-committed): EM and token-F1 on short gold answers; accuracy for verification.

Normalisation follows the SQuAD v1.1 evaluation script (lowercase, strip punctuation, drop articles
a/an/the, collapse whitespace). Scores are maxed over all gold aliases.

Per-query quality used for oracle labels and routing is always in [0, 1]:
  qa           -> token F1 (router target) and EM (reported alongside)
  verification -> accuracy (1 if predicted label == gold label)
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)
_PREFIXES = re.compile(r"^\s*(short answer|final answer|answer|verdict)\s*[:\-]\s*", re.IGNORECASE)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def extract_answer(generated: str) -> str:
    """Deterministic post-processing of raw generator output, applied identically to every config.

    Keeps the first non-empty line and strips an 'Answer:'-style prefix. Nothing else; any further
    cleaning would make EM depend on our heuristics rather than on the config.
    """
    if generated is None:
        return ""
    for line in str(generated).strip().splitlines():
        line = line.strip()
        if line:
            return _PREFIXES.sub("", line).strip()
    return ""


def exact_match(pred: str, golds: Iterable[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1_single(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        # SQuAD convention: F1 = 1 only if both empty
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, golds: Iterable[str]) -> float:
    golds = list(golds)
    return max((_f1_single(pred, g) for g in golds), default=0.0)


def answer_contains(pred: str, golds: Iterable[str]) -> float:
    """Lenient diagnostic only (never the primary metric): any normalised gold occurs in the prediction."""
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))


_LABELS = ("SUPPORTS", "REFUTES")


def parse_verdict(pred: str) -> str:
    """Map free text to SUPPORTS/REFUTES; returns 'INVALID' when neither or both appear."""
    up = pred.upper()
    hits = [lab for lab in _LABELS if lab in up or lab[:-1] in up]  # SUPPORT/REFUTE stems
    if len(hits) == 1:
        return hits[0]
    first = re.match(r"\s*([A-Z]+)", up)
    first = first.group(1) if first else ""
    if first in ("TRUE", "YES"):
        return "SUPPORTS"
    if first in ("FALSE", "NO"):
        return "REFUTES"
    return "INVALID"


def score_row(task: str, generated: str, golds: list[str]) -> dict:
    """Returns all per-query metrics for one (query, config) output. `quality` is the router target."""
    pred = extract_answer(generated)
    if task == "qa":
        em = exact_match(pred, golds)
        f1 = token_f1(pred, golds)
        return {"pred": pred, "em": em, "f1": f1, "acc": float("nan"),
                "contains": answer_contains(pred, golds), "quality": f1}
    if task == "verification":
        verdict = parse_verdict(pred)
        acc = float(verdict == golds[0].upper())
        return {"pred": verdict, "em": acc, "f1": acc, "acc": acc,
                "contains": acc, "quality": acc, "invalid": float(verdict == "INVALID")}
    raise ValueError(f"unknown task {task}")
