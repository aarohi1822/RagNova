"""Minimal chat-completion backends for answer generation and the LLM judge.

  groq   - Groq OpenAI-compatible API, model llama-3.1-8b-instant (what the
           Streamlit app uses). Needs GROQ_API_KEY.
  local  - Qwen/Qwen2.5-1.5B-Instruct via transformers, no key needed.

`auto` picks groq when GROQ_API_KEY is set, else local.
"""

from __future__ import annotations

import os
import time

GROQ_MODEL = "llama-3.1-8b-instant"
LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class GroqLLM:
    name = f"groq:{GROQ_MODEL}"

    def __init__(self) -> None:
        import requests

        self._requests = requests
        self.key = os.environ["GROQ_API_KEY"]

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        for attempt in range(5):
            resp = self._requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": GROQ_MODEL, "temperature": 0, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            time.sleep(2 ** attempt * 2)
        resp.raise_for_status()
        return ""


class LocalLLM:
    name = f"local:{LOCAL_MODEL}"

    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, torch_dtype=dtype).to(device)
        self.model.eval()
        self._torch = torch

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(new_tokens, skip_special_tokens=True).strip()


def get_llm(backend: str = "auto"):
    if backend == "auto":
        backend = "groq" if os.environ.get("GROQ_API_KEY") else "local"
    if backend == "groq":
        return GroqLLM()
    if backend == "local":
        return LocalLLM()
    raise ValueError(f"unknown backend {backend!r}")
