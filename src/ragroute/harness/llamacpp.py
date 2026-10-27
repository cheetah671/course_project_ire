"""llama.cpp backend for generation on GPUs vLLM does not support (Ada's GTX 1080 Ti, sm_61).

- `ensure_gguf`  downloads the Qwen GGUF files (official Qwen repos) into the HF cache.
- `LlamaServer`  starts llama.cpp's OpenAI-compatible `llama-server` for one model, waits for /health, stops it.
- `LlamaCppGenerator` is AutoRAG's own OpenAILLM generator module pointed at that server, with one override:
  AutoRAG's get_result re-encodes every returned token string with tiktoken (OpenAI's tokenizer) and takes
  [0], which fails on empty or partial-UTF-8 token strings from Qwen. We count output tokens from the
  server's `usage.completion_tokens` instead (exact, from llama.cpp itself) and return placeholder ids of
  that length. Everything else (prompting, batching, request parameters) is AutoRAG's code.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path


def ensure_gguf(repo: str, pattern: str) -> str:
    """Download all files matching `pattern` (split GGUFs) and return the path of the first part."""
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id=repo, allow_patterns=[pattern]))
    files = sorted(local.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matching {pattern} in {repo}")
    return str(files[0])  # llama.cpp loads "-00001-of-0000N.gguf" and finds the other parts itself


class LlamaServer:
    def __init__(self, server_bin: str, model_path: str, port: int, parallel: int, ctx_per_slot: int,
                 n_gpu_layers: int, seed: int = 0, log_path: str | None = None, lib_path: str | None = None):
        self.cmd = [server_bin, "-m", model_path, "--host", "127.0.0.1", "--port", str(port),
                    "-np", str(parallel), "-c", str(parallel * ctx_per_slot), "-ngl", str(n_gpu_layers),
                    "--seed", str(seed), "--no-webui"]
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.proc = None
        # CUDA runtime libs of the llama.cpp build env, given ONLY to the server process (putting them on the
        # global LD_LIBRARY_PATH could shadow the CUDA libs PyTorch ships with)
        self.env = dict(os.environ)
        if lib_path:
            self.env["LD_LIBRARY_PATH"] = lib_path + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")

    def __enter__(self):
        log = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, env=self.env)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self.proc.returncode}); see {self.log_path}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except Exception:  # noqa: BLE001 - server still loading
                pass
            time.sleep(5)
        raise TimeoutError("llama-server did not become healthy within 15 min")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def make_generator(project_dir: str, base_url: str, label: str, batch: int):
    """AutoRAG OpenAILLM subclass bound to a llama.cpp server (import is lazy: autorag only on the cluster)."""
    from autorag.nodes.generator.openai_llm import OpenAILLM, parse_prompt

    class LlamaCppGenerator(OpenAILLM):
        async def get_result(self, prompt, **kwargs):
            response = await self.client.chat.completions.create(
                model=self.llm, messages=parse_prompt(prompt), n=1, **kwargs)
            choice = response.choices[0]
            n_out = int(response.usage.completion_tokens) if response.usage else 0
            return choice.message.content or "", [0] * n_out, [0.0] * n_out

    return LlamaCppGenerator(project_dir, llm=label, batch=batch, base_url=f"{base_url}/v1", api_key="not-used")
