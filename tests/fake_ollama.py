"""A fake Ollama that behaves the way the real one does where the generate ladder measures it.

Read from Ollama v0.40.9 and v0.40.11 (llm/llama_server.go, envconfig/config.go) and from T4 r2's llama-server log:

- ``prompt_eval_count`` is the full length of the prompt, but llama-server reuses the longest common token prefix of
  previous prompt (``cache_prompt`` is always on), so a repeated prompt prefills about one token and its
  ``prompt_eval_duration`` is near zero.
- A prompt of ``num_ctx - 1`` tokens or more is cut to ``num_ctx - max((num_ctx - num_keep) // 2, 1)`` tokens
  (``num_keep`` is 4), and that is the count reported.
- ``OLLAMA_NUM_PARALLEL`` defaults to 1: with fewer slots than requests, the requests queue.

Time is virtual. Each request takes its prefill plus its decode, on the earliest free slot; ``clock()`` returns the
finish time of the latest request, so a pass's wall clock is exact whatever the threads do.
"""

from __future__ import annotations

import math
import threading
from typing import Any

NUM_KEEP = 4


class FakeOllama:
    def __init__(
        self,
        *,
        tokens_per_word: float = 1.0,
        prefill_tok_s: float = 4000.0,
        decode_tok_s: float = 50.0,
        cache: bool = True,
        honour_parallel: bool = True,
        report_slots: bool = True,
        long_prompt_bias: float = 0.0,
        overhead_s: float = 0.0,
        overhead_s_per_ktok: float = 0.0,
    ) -> None:
        self.tokens_per_word = tokens_per_word
        self.prefill_tok_s = prefill_tok_s
        self.decode_tok_s = decode_tok_s
        self.cache = cache
        self.honour_parallel = honour_parallel
        self.report_slots = report_slots
        self.overhead_s = overhead_s  # time that is neither prefill nor decode (Ollama's Go side)
        self.overhead_s_per_ktok = overhead_s_per_ktok
        self.long_prompt_bias = long_prompt_bias  # T4 r2: long prompts ran ~1.2% over the probes
        self.num_parallel = 1
        self.restarts = 0
        self.prompts: list[str] = []  # the timed prompts: every call that asked for more than one token
        self.contexts: list[int] = []
        self.evaluated_counts: list[int] = []
        self._lock = threading.Lock()
        self._slot_free = [0.0]
        self._now = 0.0
        self._arrival = 0.0
        self._last_words: list[str] = []
        self._loaded_ctx: int | None = None

    # -- the server surface the harness uses -------------------------------------------------------------------------
    def get(self, _path: str) -> dict[str, Any]:
        return {"models": [{"name": "m", "size": 5, "size_vram": 5}]}

    def ensure_parallel(self, slots: int) -> None:
        if slots == self.num_parallel:
            return
        self.restarts += 1
        self.num_parallel = slots if self.honour_parallel else 1
        self._slot_free = [self._now] * self.num_parallel
        self._last_words = []  # a new process has an empty cache
        self._loaded_ctx = None

    def effective_slots(self) -> int | None:
        return self.num_parallel if self.report_slots else None

    def clock(self) -> float:
        with self._lock:
            self._arrival = self._now
            return self._now

    # -- the model ---------------------------------------------------------------------------------------------------
    def tokens_of(self, prompt: str) -> int:
        words = len(prompt.split())
        bias = 1.0 + self.long_prompt_bias if words > 300 else 1.0
        return math.ceil(words * self.tokens_per_word * bias)

    def evaluated(self, payload: dict[str, Any]) -> int:
        """What Ollama reports as ``prompt_eval_count``: the prompt's tokens, or the truncated count."""

        tokens = self.tokens_of(payload["prompt"])
        window = payload["options"]["num_ctx"]
        if tokens <= window - 1:
            return tokens
        return window - max((window - NUM_KEEP) // 2, 1)

    def post(self, _path: str, payload: dict[str, Any]) -> dict[str, Any]:
        options = payload["options"]
        num_predict = int(options.get("num_predict") or 1)
        words = payload["prompt"].split()
        with self._lock:
            if self._loaded_ctx != options["num_ctx"]:
                self._loaded_ctx = options["num_ctx"]
                self._last_words = []  # a reload drops the cache
            evaluated = self.evaluated(payload)
            common = 0
            if self.cache:
                for old, new in zip(self._last_words, words, strict=False):
                    if old != new:
                        break
                    common += 1
            cached = min(math.ceil(common * self.tokens_per_word), evaluated - 1)
            self._last_words = words
            uncached = max(1, evaluated - cached)
            prefill_s = uncached / self.prefill_tok_s
            decode_s = num_predict / self.decode_tok_s
            index = min(range(len(self._slot_free)), key=self._slot_free.__getitem__)
            start = max(self._slot_free[index], self._arrival)
            overhead = self.overhead_s + self.overhead_s_per_ktok * evaluated / 1000.0
            finish = start + overhead + prefill_s + decode_s
            self._slot_free[index] = finish
            self._now = max(self._now, finish)
            total_s = finish - self._arrival  # Ollama's total_duration counts the wait for a free slot
            if num_predict > 1:
                self.prompts.append(payload["prompt"])
                self.contexts.append(options["num_ctx"])
                self.evaluated_counts.append(evaluated)
        return {
            "prompt_eval_count": evaluated,
            "prompt_eval_duration": int(prefill_s * 1e9),
            "eval_count": num_predict,
            "eval_duration": int(decode_s * 1e9),
            "load_duration": 0,
            "total_duration": int(total_s * 1e9),
        }
