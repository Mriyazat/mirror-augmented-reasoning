"""Provider-agnostic LLM client for teacher generation.

Supported back-ends (lazy-imported — only the selected one is loaded):

  - "ollama"         local Ollama server (http://localhost:11434)
                      → laptop-friendly; ideal for B1c prototype
  - "openai"         any OpenAI-compatible HTTP API
                      → works with vLLM `--served_model_name`, lm-sys
                        chat-api, Together, etc.
  - "vllm"           direct vLLM Python API (loads weights into GPU)
                      → for on-H100 generation; set VLLM_WORKER_MULTIPROC_METHOD=spawn
  - "dummy"          deterministic stub that returns a schema-valid trace
                      → unit-test / CI path; no network, no GPU

Usage:
    from src.teacher.provider import make_provider
    p = make_provider("ollama", model="llama3:70b", temperature=0.8)
    out = p.generate(system="...", user="...", n=5)
    → list[{"text": "...", "finish_reason": "stop", "prompt_tokens": int, ...}]

Every provider returns the same shape.  Retries + timeout handling live here.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Generation:
    text: str
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class Provider:
    name: str = "base"
    model: str = "none"
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 2048

    def generate(self, system: str, user: str, n: int = 1,
                 seed: int | None = None,
                 temperature_override: float | None = None) -> list[Generation]:
        """Generate `n` completions.

        `temperature_override`, when set, replaces self.temperature for
        THIS call only (used by generate.py to diversify candidates across
        temperatures without rebuilding the provider).
        """
        raise NotImplementedError

    def _effective_temperature(self, override: float | None) -> float:
        return float(override) if override is not None else float(self.temperature)


# ---------------------------------------------------------------- Dummy
class DummyProvider(Provider):
    """Schema-valid stub; no LLM call.  Used for unit tests and CI."""
    name = "dummy"

    def __init__(self, model: str = "dummy", temperature: float = 0.0, **_):
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, user: str, n: int = 1, seed: int | None = None,
                 temperature_override: float | None = None):
        _ = self._effective_temperature(temperature_override)  # accepted for API parity
        # Pull A/B DrugBank IDs out of the prompt so the "trace" is pair-specific.
        import re
        m = re.search(r"A = (\S+)\s+\((DB\d+)\).*?B = (\S+)\s+\((DB\d+)\)",
                      user, re.DOTALL)
        a_name, a_id, b_name, b_id = (m.groups() if m
                                      else ("A", "DB00001", "B", "DB00002"))
        out = []
        for i in range(n):
            body = {
                "steps": [
                    {"step_id": 1, "role": "pk_flag",
                     "claim": f"{a_name} is a CYP3A4 substrate; {b_name} inhibits CYP3A4.",
                     "evidence_ids": [a_id, b_id, "cyp3a4_inh", "cyp3a4_sub"],
                     "direction_tag": "a_to_b", "family_hint": "PK_Metabolism"},
                    {"step_id": 2, "role": "protein",
                     "claim": f"Both drugs act on shared CYP enzymes.",
                     "evidence_ids": [a_id, b_id],
                     "direction_tag": "bidirectional"},
                    {"step_id": 3, "role": "conclusion",
                     "claim": f"Metabolism of {b_name} is decreased when combined with {a_name}.",
                     "evidence_ids": [a_id, b_id], "direction_tag": "a_to_b",
                     "polarity": "down"}
                ],
                "final_answer": {
                    "family": "PK_Metabolism", "subtype": "metabolism",
                    "direction_tag": "a_to_b", "polarity": "down",
                    "confidence": 0.70 + 0.05 * i, "abstain": False,
                    "summary": f"{a_name} inhibits CYP3A4; {b_name} is a "
                               f"CYP3A4 substrate, so co-administration "
                               f"decreases {b_name} metabolism."
                }
            }
            out.append(Generation(text=json.dumps(body),
                                  finish_reason="stop",
                                  prompt_tokens=len(user) // 4,
                                  completion_tokens=200,
                                  latency_ms=10))
        return out


# ---------------------------------------------------------------- Ollama
class OllamaProvider(Provider):
    """Thin wrapper over http://localhost:11434/api/chat."""
    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b",
                 temperature: float = 0.8, top_p: float = 0.9,
                 max_tokens: int = 2048,
                 base_url: str | None = None,
                 timeout_s: int = 120, **_):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434")
        self.timeout_s = timeout_s

    def generate(self, system: str, user: str, n: int = 1,
                 seed: int | None = None,
                 temperature_override: float | None = None):
        import requests
        temp = self._effective_temperature(temperature_override)
        out = []
        url = f"{self.base_url}/api/chat"
        for i in range(n):
            t0 = time.time()
            body = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {
                    "temperature": temp,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "seed": (seed + i) if seed is not None else -1,
                },
            }
            r = requests.post(url, json=body, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            text = data.get("message", {}).get("content", "")
            out.append(Generation(
                text=text,
                finish_reason=data.get("done_reason", "stop"),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                latency_ms=int((time.time() - t0) * 1000),
            ))
        return out


# ---------------------------------------------------------------- OpenAI-compatible
class OpenAIProvider(Provider):
    """Works with any OpenAI-compatible HTTP API (vLLM --api-server, TGI, etc)."""
    name = "openai"

    def __init__(self, model: str,
                 temperature: float = 0.8, top_p: float = 0.9,
                 max_tokens: int = 2048,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 timeout_s: int = 300,
                 max_retries: int = 2, **_):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.base_url = base_url or os.environ.get(
            "OPENAI_API_BASE", "http://localhost:8000/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def generate(self, system: str, user: str, n: int = 1,
                 seed: int | None = None,
                 temperature_override: float | None = None):
        import requests
        temp = self._effective_temperature(temperature_override)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temp,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n": n,
        }
        if seed is not None:
            body["seed"] = seed

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers=headers, json=body,
                                  timeout=self.timeout_s)
                r.raise_for_status()
                data = r.json()
                usage = data.get("usage", {})
                out = []
                for ch in data.get("choices", []):
                    out.append(Generation(
                        text=ch["message"]["content"],
                        finish_reason=ch.get("finish_reason", "stop"),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_ms=int((time.time() - t0) * 1000),
                    ))
                return out
            except (requests.Timeout, requests.ConnectionError,
                    requests.HTTPError) as e:
                last_err = e
                # Exponential backoff: 2s, 4s, 8s
                if attempt < self.max_retries:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                    continue
                raise


# ---------------------------------------------------------------- vLLM direct
class VLLMProvider(Provider):
    """Direct vLLM Python API — loads weights onto the GPU.

    Must be constructed on the process that owns the CUDA devices; if running
    multiple workers, the driver should spawn each worker with its own
    VLLMProvider instance. (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    is already set by scripts/slurm/activate_env.sh.)
    """
    name = "vllm"

    def __init__(self, model: str,
                 temperature: float = 0.8, top_p: float = 0.9,
                 max_tokens: int = 2048,
                 tensor_parallel_size: int = 4,
                 dtype: str = "bfloat16",
                 gpu_memory_utilization: float = 0.90,
                 **_):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm not installed; run scripts/slurm/setup_env.sh on the "
                "cluster or pip install vllm"
            ) from e
        self._LLM = LLM
        self._SP = SamplingParams
        self._llm = LLM(model=model,
                        tensor_parallel_size=tensor_parallel_size,
                        dtype=dtype,
                        gpu_memory_utilization=gpu_memory_utilization,
                        trust_remote_code=True)

    def generate(self, system: str, user: str, n: int = 1,
                 seed: int | None = None,
                 temperature_override: float | None = None):
        temp = self._effective_temperature(temperature_override)
        # Apply chat template via the LLM's tokenizer
        tok = self._llm.get_tokenizer()
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        sp = self._SP(temperature=temp,
                      top_p=self.top_p,
                      max_tokens=self.max_tokens,
                      n=n,
                      seed=seed)
        t0 = time.time()
        outs = self._llm.generate([prompt], sp)
        latency = int((time.time() - t0) * 1000)
        # outs is a list (one RequestOutput per prompt); we sent 1 prompt
        req = outs[0]
        result = []
        for completion in req.outputs:
            result.append(Generation(
                text=completion.text,
                finish_reason=completion.finish_reason or "stop",
                prompt_tokens=len(req.prompt_token_ids),
                completion_tokens=len(completion.token_ids),
                latency_ms=latency,
            ))
        return result


# ---------------------------------------------------------------- factory
_PROVIDER_REGISTRY = {
    "dummy":  DummyProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "vllm":   VLLMProvider,
}


def make_provider(name: str, **kwargs) -> Provider:
    name = name.lower()
    if name not in _PROVIDER_REGISTRY:
        raise ValueError(f"unknown provider '{name}'; "
                         f"choose from {list(_PROVIDER_REGISTRY)}")
    return _PROVIDER_REGISTRY[name](**kwargs)
