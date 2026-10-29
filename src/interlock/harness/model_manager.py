"""Model manager for interlock's remote ladders.

Separates model loading and readiness verification from measurement probes:
- ``ensure_loaded(model, quant, ctx)`` asks the engine to load ``model`` at ``ctx`` and waits for the engine's own
  readiness signal under a size-scaled cap (``size_GiB * COLD_LOAD_S_PER_GIB * 2`` with a floor).
- ``keep(model)`` pins residency across a step.
- ``still_loaded(model)`` checks whether the model is still resident before each probe/cell.
- ``unload(model)`` unloads the model between models and on every exit path.

A load that never becomes ready within ``cap_s`` fails the step as ``load_failed`` (recording the cap and the last
engine status), never as a measurement row.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from interlock.harness.count_budget import COLD_LOAD_S_PER_GIB, GIB, MISSING_TAG_BYTES

UNRECORDED = "unrecorded"
LOAD_CAP_FLOOR_S = 180.0
OLLAMA_READY_SIGNAL = "ollama:/api/ps"
_MAX_STATIC_CLOCK_POLLS = 8


def bare_model_tag(name: object) -> str:
    """Normalize a model tag so ``foo`` and ``foo:latest`` compare equal."""

    text = name if isinstance(name, str) else ""
    return text[: -len(":latest")] if text.endswith(":latest") else text


def load_cap_s(size_bytes: int | float | None = None, *, floor_s: float = LOAD_CAP_FLOOR_S) -> float:
    """Compute the maximum seconds to wait for a model load: ``max(floor_s, size_GiB * 20.0 * 2.0)``."""

    effective_bytes = (
        float(size_bytes)
        if isinstance(size_bytes, (int, float)) and not isinstance(size_bytes, bool) and size_bytes > 0
        else float(MISSING_TAG_BYTES)
    )
    scaled = (effective_bytes / GIB) * COLD_LOAD_S_PER_GIB * 2.0
    return max(float(floor_s), scaled)


@dataclass(frozen=True)
class LoadRecord(Mapping[str, Any]):
    """Readiness and residency record returned by ``ensure_loaded``."""

    engine: str
    engine_version: str
    signal: str
    cold_load_s: float
    warm: bool
    size: int | None
    size_vram: int | None
    offload_fraction: float | None
    expires_at: str | None
    reloaded: bool
    reason: str | None = None
    model: str = UNRECORDED

    @property
    def ready_signal(self) -> str:
        return self.signal

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "signal": self.signal,
            "cold_load_s": self.cold_load_s,
            "warm": self.warm,
            "size": self.size,
            "size_vram": self.size_vram,
            "offload_fraction": self.offload_fraction,
            "expires_at": self.expires_at,
            "reloaded": self.reloaded,
            "reason": self.reason,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 11


@runtime_checkable
class ModelManager(Protocol):
    """Engine-agnostic model lifecycle interface for remote ladders."""

    def ensure_loaded(
        self,
        model: str,
        quant: str = UNRECORDED,
        ctx: int | None = None,
        *,
        size_bytes: int | None = None,
        cap_s: float | None = None,
        keep_alive: int | str = -1,
    ) -> LoadRecord: ...

    def keep(self, model: str, *, keep_alive: int | str = -1) -> None: ...

    def still_loaded(self, model: str) -> bool: ...

    def unload(self, model: str) -> None: ...


def _post_with_optional_timeout(server: Any, path: str, payload: dict[str, Any], timeout: float) -> Any:
    if "timeout" in inspect.signature(server.post).parameters:
        return server.post(path, payload, timeout=timeout)
    return server.post(path, payload)


class OllamaModelManager:
    """Model manager adapter for Ollama (readiness via empty-prompt ``/api/generate`` + ``/api/ps``)."""

    def __init__(
        self,
        server: Any,
        *,
        engine_version: str | None = None,
        clock: Callable[[], float] | None = None,
        poll_interval_s: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
        default_cap_s: float | None = None,
        size_bytes_by_model: Mapping[str, int] | None = None,
    ) -> None:
        self.server = server
        self._explicit_version = engine_version
        self._clock: Callable[[], float] = clock or getattr(server, "clock", None) or time.perf_counter
        self.poll_interval_s = poll_interval_s
        self._sleep = sleep
        self.default_cap_s = default_cap_s
        self._size_bytes_by_model: dict[str, int] = dict(size_bytes_by_model or {})
        self._ever_loaded: set[str] = set()
        self._loaded_ctx: dict[str, int | None] = {}
        self._last_record: dict[str, LoadRecord] = {}

    @property
    def engine_version(self) -> str:
        if self._explicit_version and self._explicit_version != UNRECORDED:
            return self._explicit_version
        srv_ver = getattr(self.server, "version", None)
        return str(srv_ver) if srv_ver else UNRECORDED

    def has_loaded(self, model: str) -> bool:
        """Whether ``model`` has been loaded by this manager during the current step."""

        return bare_model_tag(model) in self._ever_loaded

    def last_record(self, model: str) -> LoadRecord | None:
        """Return the most recent ``LoadRecord`` for ``model``, if any."""

        return self._last_record.get(bare_model_tag(model))

    def _query_ps(self, model: str, *, allow_single_fallback: bool = True) -> tuple[dict[str, Any] | None, str]:
        """Query ``/api/ps`` for ``model`` and return ``(entry_dict_or_none, last_status_summary)``."""

        from interlock.harness.linux_cuda import StaleCell

        try:
            ps_doc = self.server.get("/api/ps")
        except StaleCell:
            raise
        except (IndexError, KeyError, OSError, RuntimeError, ValueError, TypeError) as exc:
            return None, f"/api/ps error: {exc}"

        if not isinstance(ps_doc, Mapping):
            return None, f"/api/ps returned non-object {ps_doc!r}"
        models = ps_doc.get("models")
        if not isinstance(models, list) or not models:
            return None, "/api/ps models=[]"

        wanted = bare_model_tag(model)
        for entry in models:
            if not isinstance(entry, Mapping):
                continue
            name_match = (
                bare_model_tag(entry.get("name")) == wanted
                or bare_model_tag(entry.get("model")) == wanted
                or (allow_single_fallback and len(models) == 1)
            )
            if name_match:
                size = int(entry.get("size") or 0)
                size_vram = int(entry.get("size_vram") or 0)
                expires_at = str(entry["expires_at"]) if entry.get("expires_at") is not None else None
                if size > 0:
                    return {"size": size, "size_vram": size_vram, "expires_at": expires_at}, "resident"
                return None, f"/api/ps listed {model} with size={size}"

        listed = [str(m.get("name") or m.get("model") or "?") for m in models if isinstance(m, Mapping)]
        return None, f"/api/ps listed {listed}, not {model}"

    def _lookup_tag_size_bytes(self, model: str, explicit_size_bytes: int | None = None) -> int | None:
        if explicit_size_bytes is not None and explicit_size_bytes > 0:
            return int(explicit_size_bytes)
        bare = bare_model_tag(model)
        if model in self._size_bytes_by_model:
            return self._size_bytes_by_model[model]
        if bare in self._size_bytes_by_model:
            return self._size_bytes_by_model[bare]
        from interlock.harness.linux_cuda import StaleCell

        try:
            tags_doc = self.server.get("/api/tags")
        except StaleCell:
            raise
        except (IndexError, KeyError, OSError, RuntimeError, ValueError, TypeError):
            return None
        models = tags_doc.get("models") if isinstance(tags_doc, Mapping) else None
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, Mapping) and (
                    bare_model_tag(entry.get("name")) == bare or bare_model_tag(entry.get("model")) == bare
                ):
                    sz = int(entry.get("size") or 0)
                    if sz > 0:
                        self._size_bytes_by_model[bare] = sz
                        return sz
        return None

    def still_loaded(self, model: str) -> bool:
        """Return ``True`` when ``/api/ps`` currently lists ``model`` with ``size > 0``."""

        entry, _ = self._query_ps(model)
        return entry is not None

    def keep(self, model: str, *, keep_alive: int | str = -1) -> None:
        """Pin ``model`` residency in Ollama via ``keep_alive``."""

        payload: dict[str, Any] = {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        _post_with_optional_timeout(self.server, "/api/generate", payload, LOAD_CAP_FLOOR_S)

    def forget(self, model: str) -> None:
        """Clear residency tracking for ``model`` after a server process restart without sending an unload call."""

        bare = bare_model_tag(model)
        self._ever_loaded.discard(bare)
        self._loaded_ctx.pop(bare, None)
        self._last_record.pop(bare, None)

    def unload(self, model: str) -> None:
        """Unload ``model`` and reset per-step residency tracking for ``model``."""

        self.forget(model)
        unload_fn = getattr(self.server, "unload", None)
        if callable(unload_fn):
            unload_fn(model)
        else:
            _post_with_optional_timeout(
                self.server,
                "/api/generate",
                {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                LOAD_CAP_FLOOR_S,
            )

    def current_record(self, model: str) -> LoadRecord | None:
        """Read current ``/api/ps`` state for an already-loaded model without triggering a reload."""

        bare = bare_model_tag(model)
        entry, _ = self._query_ps(model)
        if entry is None:
            return None
        size = int(entry["size"])
        size_vram = int(entry["size_vram"])
        offload_fraction = round(size_vram / size, 6) if size > 0 else None
        prev = self._last_record.get(bare)
        return LoadRecord(
            engine="ollama",
            engine_version=self.engine_version,
            signal=OLLAMA_READY_SIGNAL,
            cold_load_s=prev.cold_load_s if prev is not None else 0.0,
            warm=True,
            size=size,
            size_vram=size_vram,
            offload_fraction=offload_fraction,
            expires_at=entry.get("expires_at"),
            reloaded=prev.reloaded if prev is not None else False,
            reason=None,
            model=model,
        )

    def ensure_loaded(
        self,
        model: str,
        quant: str = UNRECORDED,
        ctx: int | None = None,
        *,
        num_ctx: int | None = None,
        size_bytes: int | None = None,
        cap_s: float | None = None,
        keep_alive: int | str = -1,
    ) -> LoadRecord:
        """Ask Ollama to load ``model`` at ``ctx`` and wait until ``/api/ps`` confirms residency under ``cap_s``."""

        from interlock.harness.linux_cuda import (
            StaleCell,
            _failure_for,
            classify_linux_error,
        )

        if ctx is None and num_ctx is not None:
            ctx = num_ctx
        bare = bare_model_tag(model)
        ever_loaded = bare in self._ever_loaded
        pre_entry, last_status = self._query_ps(model, allow_single_fallback=ever_loaded)
        was_loaded = pre_entry is not None
        prev_ctx = self._loaded_ctx.get(bare)
        warm = was_loaded
        reloaded = bool(ever_loaded and not was_loaded)

        known_bytes = self._lookup_tag_size_bytes(
            model,
            explicit_size_bytes=size_bytes or (int(pre_entry["size"]) if pre_entry is not None else None),
        )
        if cap_s is not None:
            effective_cap_s = float(cap_s)
        elif self.default_cap_s is not None:
            effective_cap_s = float(self.default_cap_s)
        else:
            effective_cap_s = load_cap_s(known_bytes)

        needs_post = (not was_loaded) or (ctx is not None and prev_ctx != ctx)
        t0 = self._clock()
        if needs_post:
            payload: dict[str, Any] = {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
            }
            if ctx is not None:
                payload["options"] = {"num_ctx": int(ctx)}
            try:
                resp = _post_with_optional_timeout(self.server, "/api/generate", payload, effective_cap_s)
                if isinstance(resp, Mapping) and resp.get("error"):
                    raise RuntimeError(str(resp["error"]))
                if isinstance(resp, Mapping) and resp.get("done_reason"):
                    last_status = f"done_reason={resp['done_reason']}"
            except StaleCell:
                raise
            except KeyError:
                if not self.still_loaded(model):
                    raise
            except BaseException as exc:
                failure = (
                    _failure_for(self.server, exc, None)
                    if "signal: killed" in str(exc).lower()
                    else classify_linux_error(exc)
                )
                if failure.failure_class in ("oom", "load_refused", "ctx_exceeded", "generation_aborted") and (
                    "not resident" not in failure.reason.lower()
                ):
                    raise failure from exc
                elapsed_s = round(max(0.0, self._clock() - t0), 6)
                last_status = str(exc).strip() or exc.__class__.__name__
                reason = (
                    f"model {model} ({quant}) did not become ready within cap {effective_cap_s:.1f} s "
                    f"(last status: {last_status})"
                )
                failed_rec = LoadRecord(
                    engine="ollama",
                    engine_version=self.engine_version,
                    signal=OLLAMA_READY_SIGNAL,
                    cold_load_s=elapsed_s,
                    warm=warm,
                    size=None,
                    size_vram=None,
                    offload_fraction=None,
                    expires_at=None,
                    reloaded=reloaded,
                    reason=reason,
                    model=model,
                )
                raise _make_load_failed(
                    reason,
                    cap_s=effective_cap_s,
                    last_status=last_status,
                    model=model,
                    quant=quant,
                    ctx=ctx,
                    load_record=failed_rec,
                ) from exc

        static_polls = 0
        while True:
            now = self._clock()
            elapsed = max(0.0, now - t0)
            entry, status_msg = self._query_ps(model)
            last_status = status_msg
            if entry is not None and elapsed <= effective_cap_s:
                size = int(entry["size"])
                size_vram = int(entry["size_vram"])
                offload_fraction = round(size_vram / size, 6) if size > 0 else None
                record = LoadRecord(
                    engine="ollama",
                    engine_version=self.engine_version,
                    signal=OLLAMA_READY_SIGNAL,
                    cold_load_s=round(elapsed, 6),
                    warm=warm,
                    size=size,
                    size_vram=size_vram,
                    offload_fraction=offload_fraction,
                    expires_at=entry.get("expires_at"),
                    reloaded=reloaded,
                    reason=None,
                    model=model,
                )
                self._ever_loaded.add(bare)
                self._loaded_ctx[bare] = ctx
                self._last_record[bare] = record
                return record

            if elapsed >= effective_cap_s or static_polls >= _MAX_STATIC_CLOCK_POLLS:
                reason = (
                    f"model {model} ({quant}) did not become ready within cap {effective_cap_s:.1f} s "
                    f"(last status: {last_status})"
                )
                failed_rec = LoadRecord(
                    engine="ollama",
                    engine_version=self.engine_version,
                    signal=OLLAMA_READY_SIGNAL,
                    cold_load_s=round(elapsed, 6),
                    warm=warm,
                    size=None,
                    size_vram=None,
                    offload_fraction=None,
                    expires_at=None,
                    reloaded=reloaded,
                    reason=reason,
                    model=model,
                )
                raise _make_load_failed(
                    reason,
                    cap_s=effective_cap_s,
                    last_status=last_status,
                    model=model,
                    quant=quant,
                    ctx=ctx,
                    load_record=failed_rec,
                )

            before_sleep = self._clock()
            self._sleep(self.poll_interval_s)
            if self._clock() <= before_sleep:
                static_polls += 1


class LoadFailedError(Exception):
    """Base marker for a load that never became ready within ``cap_s``."""


def _make_load_failed(
    reason: str,
    *,
    cap_s: float,
    last_status: str,
    model: str = UNRECORDED,
    quant: str = UNRECORDED,
    ctx: int | None = None,
    load_record: LoadRecord | None = None,
    peak_mib: int | None = None,
) -> Any:
    from interlock.harness.linux_cuda import LinuxShapeFailure

    class _LoadFailed(LinuxShapeFailure, LoadFailedError):
        pass

    evidence: dict[str, Any] = {
        "load_cap_s": cap_s,
        "last_engine_status": last_status,
        "num_ctx_sent": ctx,
    }
    if load_record is not None:
        evidence["load_record"] = load_record.to_dict()
    err = _LoadFailed("load_failed", reason, peak_mib=peak_mib, evidence=evidence)
    err.cap_s = cap_s
    err.last_status = last_status
    err.model = model
    err.quant = quant
    err.ctx = ctx
    err.load_record = load_record
    return err
