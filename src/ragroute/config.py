"""Experiment config loading and the 24-config grid.

`load_experiment()` reads configs/experiment.yaml and expands ${VAR:-default} references.
`all_configs()` enumerates the fully crossed grid; `RagConfig.config_id` is the key used everywhere.
"""
from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def load_experiment(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "experiment.yaml"
    cfg = load_yaml(path)
    # resolve relative paths against the repo root
    for key, val in cfg["paths"].items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else REPO_ROOT / p)
    tsv = Path(cfg["corpus"]["tsv_path"])
    cfg["corpus"]["tsv_path"] = str(tsv if tsv.is_absolute() else REPO_ROOT / tsv)
    if os.environ.get("RAGROUTE_PILOT") == "1":
        p = cfg.get("pilot", {})
        cfg["corpus"]["subcorpus_size"] = p.get("subcorpus_size", cfg["corpus"]["subcorpus_size"])
        for d in cfg["datasets"].values():
            d["n"] = p.get("n_per_dataset", d["n"])
        cfg["is_pilot"] = True
    return cfg


def load_cost_weights(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "cost_weights.yaml"
    return load_yaml(path)


def enabled_datasets(cfg: dict) -> list[str]:
    return [name for name, d in cfg["datasets"].items() if d.get("enabled", False)]


@dataclass(frozen=True, order=True)
class RagConfig:
    retriever: str      # bm25 | dense | hybrid
    top_k: int          # passages given to the generator
    rerank: bool        # cross-encoder on a fixed pool, then truncate to top_k
    generator: str      # key into cfg["models"]["generators"]

    @property
    def config_id(self) -> str:
        return f"{self.retriever}-k{self.top_k}-rr{int(self.rerank)}-{self.generator}"

    @property
    def retrieval_key(self) -> str:
        """Configs sharing this key share the same passages (generator-independent)."""
        return f"{self.retriever}-k{self.top_k}-rr{int(self.rerank)}"

    @classmethod
    def from_id(cls, config_id: str) -> "RagConfig":
        retriever, k, rr, gen = config_id.split("-", 3)
        return cls(retriever, int(k[1:]), bool(int(rr[2:])), gen)


def all_configs(cfg: dict) -> list[RagConfig]:
    g = cfg["grid"]
    return [
        RagConfig(r, int(k), bool(rr), gen)
        for r, k, rr, gen in itertools.product(g["retrievers"], g["top_k"], g["reranker"], g["generators"])
    ]


def config_feature_frame(configs: list[RagConfig]):
    """One row per config with the encoded config features used by the router (feature group 3)."""
    import pandas as pd

    rows = []
    for c in configs:
        rows.append(
            {
                "config_id": c.config_id,
                "cfg_ret_bm25": int(c.retriever == "bm25"),
                "cfg_ret_dense": int(c.retriever == "dense"),
                "cfg_ret_hybrid": int(c.retriever == "hybrid"),
                "cfg_topk": c.top_k,
                "cfg_rerank": int(c.rerank),
                "cfg_gen_large": int("7b" in c.generator),
            }
        )
    return pd.DataFrame(rows)
