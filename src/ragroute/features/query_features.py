"""Feature groups 1 (query) and 2 (data). Everything here is available BEFORE retrieval.

Group 1, query: surface + cue features from the text, and the question embedding (same bge model the
dense retriever uses; reduced by PCA fitted on train inside the router pipeline). The learned
"multi-hop score" is produced in router/train_eval.py (out-of-fold on train) because it needs labels.

Group 2, data: dataset identity and dataset-level attributes, so one router serves several datasets.
Caveat for the write-up: on our data, "multi-hop" is almost a function of dataset identity (HotpotQA),
so the multi-hop score and dataset one-hots are partly redundant; ablations separate them.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

WH = ["what", "who", "whom", "whose", "when", "where", "which", "why", "how"]
AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had", "will", "would", "should"}
COMPARISON = re.compile(r"\b(both|either|neither|same|than|more|less|fewer|older|younger|earlier|later|first|"
                        r"larger|smaller|longer|shorter|higher|lower|bigger|greater)\b", re.I)
OR_CHOICE = re.compile(r"\b\w+\s+or\s+\w+\b", re.I)
REL_PRON = re.compile(r"(?<!^)\b(who|which|that|whose|where)\b", re.I)
OF_THE = re.compile(r"\bof the\b", re.I)
POSSESSIVE = re.compile(r"\w's\b")
CAP_SPAN = re.compile(r"(?:[A-Z][\w\-'.]*)(?:\s+(?:of|the|de|and|&)?\s*[A-Z][\w\-'.]*)*")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")

DATASET_ATTRS = {
    # dataset-level attributes (domain-level, not per-query): multi-hop?, verification task?, trivia style?
    "nq": {"ds_multihop": 0, "ds_verification": 0, "ds_trivia_style": 0},
    "hotpotqa": {"ds_multihop": 1, "ds_verification": 0, "ds_trivia_style": 0},
    "triviaqa": {"ds_multihop": 0, "ds_verification": 0, "ds_trivia_style": 1},
    "fever": {"ds_multihop": 0, "ds_verification": 1, "ds_trivia_style": 0},
}


def entity_spans(text: str) -> list[str]:
    """Capitalised spans, excluding a sentence-initial single word (heuristic; spaCy NER is an optional upgrade)."""
    spans = [m.group(0) for m in CAP_SPAN.finditer(text)]
    if spans and text.startswith(spans[0]) and " " not in spans[0]:
        spans = spans[1:]
    return spans


def query_surface_features(query: str) -> dict:
    q = query.strip()
    toks = q.split()
    first = toks[0].lower() if toks else ""
    lower = q.lower()
    f = {
        "q_len_tokens": len(toks),
        "q_len_chars": len(q),
        "q_is_question": int(q.endswith("?") or first in WH or first in AUX),
        "q_starts_aux": int(first in AUX),
        "q_n_entities": len(entity_spans(q)),
        "q_n_years": len(YEAR.findall(q)),
        "q_n_numbers": len(NUM.findall(q)),
        "q_n_quotes": q.count('"') // 2,
        "q_comparison_cues": len(COMPARISON.findall(q)),
        "q_or_choice": int(bool(OR_CHOICE.search(q))),
        "q_rel_pronouns": len(REL_PRON.findall(q)),
        "q_of_the": len(OF_THE.findall(q)),
        "q_possessive": len(POSSESSIVE.findall(q)),
    }
    wh_hit = next((w for w in WH if re.search(rf"\b{w}\b", lower)), "none")
    for w in WH + ["none"]:
        f[f"q_wh_{w}"] = int(wh_hit == w)
    # bridge cue: nested description of an entity (relative clause or 'of the' chain) - HotpotQA bridge style
    f["q_bridge_cues"] = f["q_rel_pronouns"] + f["q_of_the"] + f["q_possessive"]
    return f


def data_features(dataset: str, all_datasets: list[str]) -> dict:
    f = {f"ds_is_{d}": int(dataset == d) for d in all_datasets}
    f.update(DATASET_ATTRS.get(dataset, {"ds_multihop": 0, "ds_verification": 0, "ds_trivia_style": 0}))
    return f


def build_query_features(meta: pd.DataFrame) -> pd.DataFrame:
    datasets = sorted(meta["dataset"].unique())
    rows = []
    for qid, q, ds in zip(meta["qid"], meta["query"], meta["dataset"]):
        rows.append({"qid": qid, **query_surface_features(q), **data_features(ds, datasets)})
    return pd.DataFrame(rows)


def embed_questions(queries: list[str], model_name: str, instruction: str, device: str | None = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode([instruction + q for q in queries], batch_size=128, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)
