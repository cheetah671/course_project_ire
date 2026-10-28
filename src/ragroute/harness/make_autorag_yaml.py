"""Generate AutoRAG YAML files (verified against AutoRAG 0.3.27 source).

1. configs/autorag/retrieval_stage.yaml
   lexical_retrieval(bm25, top_k=pool) -> semantic_retrieval(vectordb, top_k=pool)
   -> hybrid_retrieval(hybrid_rrf, weight_range=(rrf_k, rrf_k), top_k=pool)
   One trial; AutoRAG saves every module's output ({node}/0.parquet) so all three pools are cached.

2. configs/autorag/full/<config_id>.yaml  (24 files, one module per node = no selection, no pruning)
   Used (a) as the human-readable definition of each config and (b) by check_equivalence.py, which runs
   2 of them end-to-end through AutoRAG's Evaluator on a small subset to prove the staged runner
   reproduces AutoRAG's own pipeline.

AutoRAG facts this relies on (0.3.27):
 * Each node keeps only its best module for the next node -> with >1 module per node the grid is pruned.
   Hence one module per node.
 * hybrid_rrf grid-searches `weight` (the RRF constant) over weight_range (default (4, 80)) on the eval
   set; pinning weight_range to (60, 60) fixes k=60 and avoids tuning on test queries.
 * hybrid_retrieval consumes *_lexical and *_semantic columns, so it must follow both nodes.
 * Downstream nodes read retrieval columns in the order: unsuffixed (hybrid) > _semantic > _lexical.
 * `vectordb: default` means OpenAI embeddings -> we always name our own vectordb.
 * A prompt_maker node with a single module skips its (LLM-based) evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ragroute.config import all_configs, load_experiment

RET_METRICS = ["retrieval_recall", "retrieval_ndcg", "retrieval_mrr"]
GEN_METRICS = ["bleu", "rouge"]  # cheap; our EM/F1 are computed outside AutoRAG (AutoRAG has no EM/F1)
VDB_NAME = "bge_base_chroma"


def vectordb_block(cfg: dict) -> list:
    m = cfg["models"]
    return [{
        "name": VDB_NAME,
        "db_type": "chroma",
        "client_type": "persistent",
        "embedding_model": [{
            "type": "huggingface",
            "model_name": m["embedding"],
            "query_instruction": m["embedding_query_instruction"],
            "embed_batch_size": 256,
        }],
        "collection_name": "bge_base",
        "path": str(Path(cfg["paths"]["project_dir"]) / "resources" / "chroma"),
        "embedding_batch": 256,
        "similarity_metric": "cosine",
    }]


def _lexical(top_k, cfg):
    return {"node_type": "lexical_retrieval", "strategy": {"metrics": RET_METRICS}, "top_k": top_k,
            "modules": [{"module_type": "bm25", "bm25_tokenizer": cfg["models"]["bm25_tokenizer"]}]}


def _semantic(top_k):
    return {"node_type": "semantic_retrieval", "strategy": {"metrics": RET_METRICS}, "top_k": top_k,
            "modules": [{"module_type": "vectordb", "vectordb": VDB_NAME}]}


def _hybrid(top_k, cfg):
    k = cfg["grid"]["rrf_k"]
    return {"node_type": "hybrid_retrieval", "strategy": {"metrics": RET_METRICS}, "top_k": top_k,
            "modules": [{"module_type": "hybrid_rrf", "weight_range": f"({k}, {k})"}]}


def retrieval_nodes(retriever: str, depth: int, cfg: dict) -> list:
    pool = cfg["grid"]["hybrid_pool"]
    if retriever == "bm25":
        return [_lexical(depth, cfg)]
    if retriever == "dense":
        return [_semantic(depth)]
    if retriever == "hybrid":
        return [_lexical(pool, cfg), _semantic(pool), _hybrid(depth, cfg)]
    raise ValueError(retriever)


def prompt_node(cfg: dict, task: str = "qa") -> dict:
    return {"node_type": "prompt_maker", "strategy": {"metrics": GEN_METRICS},
            "modules": [{"module_type": "chat_fstring", "prompt": [cfg["prompts"][task]]}]}


def generator_node(cfg: dict, gen_key: str) -> dict:
    g = cfg["models"]["generators"][gen_key]
    gen = cfg["generation"]
    if gen.get("backend") == "llamacpp":
        # AutoRAG's openai_llm module pointed at a running llama.cpp server for this generator.
        # `llm` is only a label AutoRAG accepts; the server serves the Qwen GGUF it was started with.
        lc = gen["llamacpp"]
        return {"node_type": "generator", "strategy": {"metrics": GEN_METRICS},
                "modules": [{
                    "module_type": "openai_llm", "llm": lc["autorag_model_label"],
                    "base_url": f"http://127.0.0.1:{lc['port']}/v1", "api_key": "not-used",
                    "temperature": gen["temperature"], "max_tokens": gen["max_tokens"], "seed": gen["seed"],
                    "batch": lc["parallel"] * 2,
                }]}
    return {"node_type": "generator", "strategy": {"metrics": GEN_METRICS},
            "modules": [{
                "module_type": "vllm", "llm": g.get("vllm_model", g["hf_id"]),
                "temperature": gen["temperature"], "max_tokens": gen["max_tokens"],
                "tensor_parallel_size": g["tensor_parallel_size"], "dtype": gen["dtype"],
                "max_model_len": gen["max_model_len"],
                "gpu_memory_utilization": gen["gpu_memory_utilization"], "max_num_seqs": gen["max_num_seqs"],
                "max_num_batched_tokens": gen["max_num_batched_tokens"],
                "seed": gen["seed"],
            }]}


def retrieval_stage_yaml(cfg: dict) -> dict:
    pool = cfg["grid"]["rerank_pool"]
    return {"vectordb": vectordb_block(cfg),
            "node_lines": [{"node_line_name": "retrieve_node_line",
                            "nodes": [_lexical(pool, cfg), _semantic(pool), _hybrid(pool, cfg)]}]}


def full_config_yaml(cfg: dict, c, task: str = "qa") -> dict:
    pool = cfg["grid"]["rerank_pool"]
    depth = pool if c.rerank else c.top_k
    post = []
    if c.rerank:
        post.append({"node_type": "passage_reranker", "strategy": {"metrics": RET_METRICS}, "top_k": c.top_k,
                     "modules": [{"module_type": "sentence_transformer_reranker",
                                  "model_name": cfg["models"]["reranker"], "batch": 64}]})
    post += [prompt_node(cfg, task), generator_node(cfg, c.generator)]
    return {"vectordb": vectordb_block(cfg),
            "node_lines": [
                {"node_line_name": "retrieve_node_line", "nodes": retrieval_nodes(c.retriever, depth, cfg)},
                {"node_line_name": "post_retrieve_node_line", "nodes": post},
            ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default: paths.autorag_yaml_dir from the config")
    args = ap.parse_args()
    cfg = load_experiment(args.config)
    out = Path(args.out or cfg["paths"]["autorag_yaml_dir"])
    (out / "full").mkdir(parents=True, exist_ok=True)
    class NoAliasDumper(yaml.SafeDumper):  # no &id001 anchors: every node gets its own metrics list
        def ignore_aliases(self, data):
            return True

    dump = lambda d, p: p.write_text(yaml.dump(d, Dumper=NoAliasDumper, sort_keys=False, allow_unicode=True))  # noqa: E731
    dump(retrieval_stage_yaml(cfg), out / "retrieval_stage.yaml")
    for c in all_configs(cfg):
        dump(full_config_yaml(cfg, c), out / "full" / f"{c.config_id}.yaml")
    print(f"wrote retrieval_stage.yaml and {len(all_configs(cfg))} full-config YAMLs to {out}")


if __name__ == "__main__":
    main()
