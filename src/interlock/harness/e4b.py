"""E4b's run: its own Ollama, per-slot warm-up, refusals, and the pilot.

The spec is the contract (experiments/e4b/spec.yaml). This module enforces the
rules the loader used to ignore. Tests pass fakes. Nothing here opens a GPU
or a shared Ollama port unless a caller builds the live seams and runs them.

The 5090 companion is a generate loop. A paired NPU or iGPU slot waits until
that loop has finished one steady warm decode and the next decode is underway,
then runs the Intel warm and measure. Stopping the loop does not unload
qwen3. The model loads once at block start and unloads once, with keep_alive 0,
at block end. An unload in the middle would reload it and refuse the next slot.

The run-owned server inherits the process environment with every OLLAMA_* and
LLAMA_ARG_* key removed, then the recorded host, models directory and debug
flag. Before it starts, the watchdog heartbeat and an empty card are required.
A dirty worktree at the start refuses before any device starts. After the
first load, the run refuses unless this start's log names cuda_v13,
size_vram equals size, and the digest matches the spec pin. Flash attention
and the KV cache type on the header come from that load. The environment
value is kept as config_flash_attention. A missing load line refuses before
any measured slot.

Live seams sample the GPU through NVML, then nvidia-smi, and the CPU through
typeperf. The burn's threads stop when the burn returns, including on abort.
Probes time out after 10 s and a generate after 300 s. A pilot call with no
within-call clock range is missing; the pilot needs at least 18 good calls.
The prompt is a counting prefix, not E4's repeated sentence. A token-repeat
500 is a short decode. Each call records how its time was spent.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import threading
import time
import traceback
from pathlib import Path

from interlock.guards import require_card_empty, require_grant, require_heartbeat, require_npu_session
from interlock.harness.e4b_stats import bootstrap_median, decide, effect
from interlock.harness.lifecycle import move_aside, refuse_strays, require_free, resolve_run_id, utc_now
from interlock.harness.ollama_serve import (
    HOST,
    PORT,
    SHARED_PORTS,
    assert_port_free,
    get_json,
    read_log_segment,
    receipt_environment,
    resolve_pin,
    stop_process_tree,
)
from interlock.harness.timing import percentile


class E4bError(Exception):
    """E4b cannot run this spec, or a slot broke a rule the receipt must show."""


class RepeatLimit(E4bError):
    """Ollama HTTP 500 said token repeat limit. A short decode, not a dead run."""

    def __init__(self, message: str, time_spent: dict | None = None):
        super().__init__(message)
        self.time_spent = time_spent


PILOT_CALLS = 20
PILOT_MIN_GOOD = 18
PILOT_CLOCK_RULE = (
    "A measured call with no within-call clock range is missing and does not abort the pilot. "
    "The pilot needs at least 18 good calls; fewer is a refusal."
)
PROBE_TIMEOUT_S = 10
GENERATE_TIMEOUT_S = 300
CPU_UTILITY = r"\Processor Information(_Total)\% Processor Utility"
CPU_PERFORMANCE = r"\Processor Information(_Total)\% Processor Performance"
LOAD_REFUSE_S = 1.0
CLOCK_STEP_MHZ = 7.5
OWNED = f"http://{HOST}:{PORT}"


def round_up_mhz(mhz: float, *, step: float = CLOCK_STEP_MHZ) -> float:
    """Smallest multiple of ``step`` that is still at least ``mhz``."""

    if mhz < 0:
        raise E4bError("a clock range cannot be negative")
    if step <= 0:
        raise E4bError("the clock step must be positive")
    return math.ceil(mhz / step - 1e-12) * step


def is_steady(samples: list[float] | None, *, band: float | None, count: int) -> bool:
    """True when the last ``count`` samples all sit within ``band`` of the maximum."""

    if band is None or band < 0 or not samples or len(samples) < count:
        return False
    if any(sample is None for sample in samples):
        return False
    peak = max(samples)
    return all(abs(peak - sample) <= band + 1e-9 for sample in samples[-count:])


def parse_nvidia_smi(text: str) -> tuple[float | None, float | None]:
    """``clocks.sm, power.draw`` from one csv row. ``[N/A]`` is a missing reading."""

    line = ""
    for raw in text.splitlines():
        if raw.strip():
            line = raw.strip()
    if not line or "," not in line:
        return None, None
    left, right = line.split(",", 1)
    return _first_number(left), _first_number(right)


def parse_typeperf(text: str) -> float | None:
    """The last numeric field of a typeperf CSV. The header line is skipped."""

    found = None
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("("):
            continue
        for field in reversed(raw.split(",")):
            field = field.strip().strip('"')
            try:
                found = float(field)
                break
            except ValueError:
                continue
    return found


def _first_number(token: str) -> float | None:
    token = token.strip()
    if not token or "N/A" in token.upper():
        return None
    number = []
    started = False
    for char in token:
        if char.isdigit() or char == "." or (char == "-" and not started):
            number.append(char)
            started = True
        elif started:
            break
    if not number or number == ["-"] or number == ["."]:
        return None
    try:
        return float("".join(number))
    except ValueError:
        return None


def sample_gpu(*, nvml=None, smi=None) -> tuple[float | None, float | None]:
    """NVML first. A failure falls through to nvidia-smi. Neither tool is imported here."""

    if nvml is not None:
        try:
            clock, power = nvml()
            return clock, power
        except Exception:  # noqa: S110  # NVML failure falls through to nvidia-smi
            pass
    if smi is None:
        return None, None
    try:
        text = smi()
    except Exception:
        return None, None
    return parse_nvidia_smi(text if isinstance(text, str) else "")


def read_nvml(library) -> tuple[float, float]:
    """SM clock in MHz and power in watts. NVML reports power in milliwatts."""

    library.nvmlInit()
    handle = library.nvmlDeviceGetHandleByIndex(0)
    clock = library.nvmlDeviceGetClockInfo(handle, library.NVML_CLOCK_SM)
    power_mw = library.nvmlDeviceGetPowerUsage(handle)
    return float(clock), float(power_mw) / 1000.0


def smi_query(run) -> str:
    """The csv query the spec names. ``run`` returns a completed process or the text."""

    completed = run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"])
    if isinstance(completed, str):
        return completed
    stdout = getattr(completed, "stdout", "")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    return stdout or ""


def typeperf_once(counter: str, run) -> float | None:
    completed = run(["typeperf", counter, "-sc", "1"])
    if isinstance(completed, str):
        return parse_typeperf(completed)
    stdout = getattr(completed, "stdout", "")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    return parse_typeperf(stdout or "")


class SampleRecord:
    """Which GPU probe answered, and the gaps between samples in this run."""

    def __init__(self) -> None:
        self.probe: str | None = None
        self.intervals: list[float] = []


def live_sample_gpu(record: SampleRecord | None = None) -> tuple[float | None, float | None]:
    """NVML first, then nvidia-smi. The record keeps whichever probe returned."""

    def nvml():
        import pynvml

        clock, power = read_nvml(pynvml)
        if record is not None:
            record.probe = "nvml"
        return clock, power

    def smi():
        text = smi_query(capture_run)
        if record is not None:
            record.probe = "smi"
        return text

    return sample_gpu(nvml=nvml, smi=smi)


class LiveGpuSample:
    """The live ``sample_gpu`` seam. Calling it is a real probe, so tests do not."""

    live = True

    def __init__(self, record: SampleRecord) -> None:
        self.record = record

    def __call__(self) -> tuple[float | None, float | None]:
        return live_sample_gpu(self.record)


def live_sample_cpu(name: str) -> float | None:
    """typeperf for % Processor Utility and % Processor Performance."""

    if name == "utility":
        counter = CPU_UTILITY
    elif name == "performance":
        counter = CPU_PERFORMANCE
    else:
        return None
    return typeperf_once(counter, capture_run)


live_sample_cpu.live = True


class CpuBurn:
    """Busy threads. ``stop`` joins them, including when the burn raises."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return
        for _ in range(count):
            thread = threading.Thread(target=self._spin, name="interlock-e4b-burn", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        threads = list(self._threads)
        self._threads.clear()
        for thread in threads:
            thread.join(timeout=2)
        self._stop.clear()

    def _spin(self) -> None:
        while not self._stop.is_set():
            for _ in range(1000):
                if self._stop.is_set():
                    return


def _stop_burn(start_threads) -> None:
    """Stop a burn that published ``stop``, or a bound :meth:`CpuBurn.start`."""

    stop = getattr(start_threads, "stop", None)
    if not callable(stop):
        owner = getattr(start_threads, "__self__", None)
        stop = getattr(owner, "stop", None) if owner is not None else None
    if callable(stop):
        stop()


def collect_series(*, sample, sleep, every_s: float, duration_s: float) -> list[tuple]:
    """Sample at the start and every ``every_s`` until ``duration_s`` has elapsed."""

    if every_s <= 0:
        raise E4bError("the sample interval must be positive")
    if duration_s < 0:
        raise E4bError("a call duration cannot be negative")
    series = []
    elapsed = 0.0
    while True:
        series.append(sample())
        if elapsed + 1e-12 >= duration_s:
            break
        step = min(every_s, duration_s - elapsed)
        sleep(step)
        elapsed += step
    return series


def refusal_reason(
    device: str,
    result: dict,
    *,
    num_predict: int | None,
    clock_tolerance: float | None,
    steady_ok: bool | None,
) -> str | None:
    """The slot's refusal, or None when the call may enter the estimate."""

    load = result.get("load_duration_s")
    if load is not None and load > LOAD_REFUSE_S:
        return "load_duration_s"
    if device == "5090":
        if steady_ok is False:
            return "clock_not_steady"
        clocks = result.get("clocks_mhz")
        power = result.get("power_w")
        if not isinstance(clocks, list) or not clocks or any(item is None for item in clocks):
            return "missing_clock"
        if not isinstance(power, list) or not power or any(item is None for item in power):
            return "missing_power"
        if clock_tolerance is not None and (max(clocks) - min(clocks)) > clock_tolerance:
            return "clock_range"
        count = result.get("eval_count")
        if num_predict is not None and (isinstance(count, bool) or not isinstance(count, int) or count < num_predict):
            return "short_decode"
        return None
    if device in {"NPU", "iGPU"}:
        if result.get("truncated_tokens") not in (0, 1):
            return "truncated_tokens"
    return None


def slot_refusal(arm: dict, result: dict, *, companion_arm: dict | None, companion: dict | None, tolerance, steady_ok,
    num_predict) -> str | None:
    """Measured call first, then the companion under the same rules."""

    reason = refusal_reason(
        arm["device"],
        result,
        num_predict=num_predict,
        clock_tolerance=tolerance,
        steady_ok=steady_ok,
    )
    if reason or companion_arm is None or companion is None:
        return reason
    companion_reason = refusal_reason(
        companion_arm["device"],
        companion,
        num_predict=companion_arm.get("num_predict"),
        clock_tolerance=tolerance,
        steady_ok=None,
    )
    if companion_reason:
        return "companion_" + companion_reason
    return None


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def call_time_spent(result: dict | None, *, kind: str, reason: str | None = None) -> dict:
    """Where one call spent its time, in seconds.

    Ollama reports nanoseconds. A result that already converted them keeps the
    seconds. A missing slice stays null. ``kind`` is warm, measure, or
    companion, so a warm-up that dies is not mistaken for the measured decode.
    """

    result = result if isinstance(result, dict) else {}
    nested = result.get("time_spent") if isinstance(result.get("time_spent"), dict) else {}

    def seconds(second_keys: tuple[str, ...], nanosecond_key: str) -> float | None:
        for key in second_keys:
            number = _number(result.get(key))
            if number is not None:
                return number
        nanoseconds = _number(result.get(nanosecond_key))
        if nanoseconds is not None:
            return nanoseconds / 1e9
        for key in second_keys:
            number = _number(nested.get(key))
            if number is not None:
                return number
        return None

    eval_count = result.get("eval_count")
    prompt_count = result.get("prompt_eval_count")
    spent = {
        "kind": kind,
        "load_s": seconds(("load_s", "load_duration_s"), "load_duration"),
        "prompt_eval_s": seconds(("prompt_eval_s",), "prompt_eval_duration"),
        "eval_s": seconds(("eval_s", "elapsed_s"), "eval_duration"),
        "total_s": seconds(("total_s",), "total_duration"),
        "eval_count": eval_count if isinstance(eval_count, int) and not isinstance(eval_count, bool) else None,
        "prompt_eval_count": prompt_count if isinstance(prompt_count, int) and not isinstance(prompt_count,
            bool) else None,
    }
    if reason:
        spent["reason"] = reason
    return spent


def _is_repeat_limit(exc: BaseException) -> bool:
    if isinstance(exc, RepeatLimit):
        return True
    return "token repeat limit" in str(exc).lower()


def _repeat_spent(exc: BaseException, kind: str) -> dict:
    spent = None
    if isinstance(exc, RepeatLimit) and isinstance(exc.time_spent, dict):
        spent = dict(exc.time_spent)
    if spent is None:
        spent = call_time_spent({}, kind=kind, reason="token repeat limit")
    spent["kind"] = kind
    spent["reason"] = "token repeat limit"
    return spent


def _json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def counting_prompt(target_tokens: int, count_tokens) -> str:
    """Shortest line-prefix of ``1\\n2\\n3\\n...`` whose count is at least the target.

    The prefix one line shorter counts below the target. The cut is a whole
    line, not a repeated seed.
    """

    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens < 1:
        raise E4bError("E4b fits its prompt from the dataset seed and target")
    if count_tokens is None:
        raise E4bError("E4b fits the prompt with the model tokenizer")
    text = ""
    number = 1
    limit = max(target_tokens * 100, 10_000)
    while number <= limit:
        text = f"{text}{number}\n"
        count = count_tokens(text)
        if isinstance(count, bool) or not isinstance(count, int):
            raise E4bError("the token count must be an integer")
        if count >= target_tokens:
            return text
        number += 1
    raise E4bError("the counting prompt never reaches the token target")


def prepare_prompt(spec: dict, count_tokens) -> str:
    """One counting prefix for every arm. The counter is the model tokenizer."""

    dataset = spec.get("dataset") or {}
    target = dataset.get("target_tokens")
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise E4bError("E4b fits its prompt from the dataset seed and target")
    if count_tokens is None:
        raise E4bError("E4b fits the prompt with the model tokenizer")
    return counting_prompt(target, count_tokens)


def capture_run(args: list[str]):
    """The live runner: a bare ``subprocess.run`` returns no stdout, so the pin and git reads saw ""."""

    return subprocess.run(args, capture_output=True, text=True, check=False)


def read_git(repo: Path, run) -> dict:
    """HEAD and whether the worktree is clean. The receipt stores the sha, not a path.

    Fails closed: both git calls must succeed and HEAD must be a 40-hex sha, and the
    tree is clean only on a successful, empty porcelain read. A runner that captured
    nothing used to give ``git_sha ""`` and ``clean_tree True``.
    """

    def _out(args: list[str]) -> str:
        completed = run(args)
        if isinstance(completed, str):
            return completed
        code = getattr(completed, "returncode", 0)
        if code != 0:
            raise E4bError(f"git {args[3]} failed with exit code {code}")
        stdout = getattr(completed, "stdout", None)
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if stdout is None:
            raise E4bError(f"git {args[3]} returned no output (the runner captured nothing)")
        return stdout

    sha = _out(["git", "-C", repo.as_posix(), "rev-parse", "HEAD"]).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise E4bError(f"git rev-parse HEAD did not return a 40-hex sha ({sha!r})")
    porcelain = _out(["git", "-C", repo.as_posix(), "status", "--porcelain"])
    lines = [line for line in porcelain.splitlines() if line.strip()]
    return {"git_sha": sha, "clean_tree": not lines, "porcelain": lines}


# A home directory, not a particular account. The reason must not carry one.
_HOME_PREFIX = re.compile(
    r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+|[\\/]Users[\\/]+[^\\/]+|[\\/]home[\\/]+[^\\/]+)"
)


def _public_status_line(line: str) -> str:
    return _HOME_PREFIX.sub("~", line)


def refuse_dirty_worktree(recorded: dict | None) -> None:
    """A dirty tree at the start is a refusal. A clean tree, or no read, is not."""

    if not isinstance(recorded, dict) or recorded.get("clean_tree") is not False:
        return
    raw = recorded.get("porcelain")
    lines = [line for line in raw if isinstance(line, str) and line.strip()] if isinstance(raw, list) else []
    shown = [_public_status_line(line) for line in lines[:3]]
    if len(lines) > 3:
        shown.append(f"and {len(lines) - 3} more")
    detail = "; ".join(shown) if shown else "git status --porcelain was not empty"
    raise E4bError(f"REFUSED: the worktree is dirty at the start ({detail})")


def _flash_flag(state: str | None) -> str | None:
    """The load line says enabled or disabled. The header keeps true or false."""

    if state == "enabled":
        return "true"
    if state == "disabled":
        return "false"
    return None


def _load_facts(segment: str) -> dict:
    from interlock.harness.fill import load_line_facts

    return load_line_facts(segment)


def _require_load_lines(seams, running) -> dict:
    """The same first-load reader E5 uses. A missing line refuses the slot."""

    facts = _load_facts(_log_text(seams, running))
    reason = facts.get("reason") if isinstance(facts, dict) else "the first load did not record flash attention"
    if reason:
        raise E4bError("REFUSED: " + str(reason))
    return facts


def card_models_at_ceiling(read_mib, *, ceiling: int) -> list:
    """A one-item list when the card is above the idle ceiling or cannot be read.

    The list is what :func:`require_card_empty` refuses. Nothing here opens
    11434 or 11490. ``vram_mib_at_start`` stays a recording, not this gate.
    """

    reading = read_mib()
    if isinstance(reading, bool) or not isinstance(reading, int) or reading > ceiling:
        return [reading]
    return []


def ollama_pin(spec: dict) -> tuple[str, str]:
    """The 5090 model's name and the spec's digest pin."""

    chosen = None
    for arm in spec.get("arms") or []:
        if not isinstance(arm, dict) or arm.get("device") != "5090":
            continue
        if arm.get("id") == "5090-alone":
            chosen = arm
            break
        if chosen is None:
            chosen = arm
    if chosen is None:
        raise E4bError("E4b has no 5090 arm to pin")
    model = chosen.get("model")
    digest = chosen.get("revision_or_digest")
    if not isinstance(model, str) or not model:
        raise E4bError("the 5090 arm has no model")
    if not isinstance(digest, str) or not digest.strip() or digest.strip() == "unrecorded":
        raise E4bError("the 5090 arm has no digest pin")
    return model, digest.strip()


def _digest_hex(value) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip().lower()
    prefix = "sha256:"
    if token.startswith(prefix):
        token = token[len(prefix):]
    return token


def segment_names_cuda_v13(segment: str) -> bool:
    """True when a ``load_backend`` line in this start's segment names ``cuda_v13``."""

    for line in segment.splitlines():
        if "load_backend" in line and "cuda_v13" in line:
            return True
    return False


def _ps_entry(ps: dict, model: str) -> dict | None:
    models = ps.get("models") if isinstance(ps, dict) else None
    if not isinstance(models, list):
        return None
    for entry in models:
        if isinstance(entry, dict) and (entry.get("name") == model or entry.get("model") == model):
            return entry
    return None


def prove_owned_server(spec: dict, *, segment: str, ps: dict) -> None:
    """Refuse unless this start is cuda_v13, fully resident, and the pinned digest."""

    if not segment_names_cuda_v13(segment):
        raise E4bError("REFUSED: the server log does not show cuda_v13 in its load_backend line")
    model, pin = ollama_pin(spec)
    entry = _ps_entry(ps, model)
    size = None if entry is None else entry.get("size")
    vram = None if entry is None else entry.get("size_vram")
    if isinstance(size, bool) or isinstance(vram, bool) or not isinstance(size, int) or not isinstance(vram,
        int) or vram != size:
        raise E4bError("REFUSED: the first load is not fully resident; size_vram must equal size")
    if _digest_hex(entry.get("digest")) != _digest_hex(pin):
        raise E4bError("REFUSED: the model digest does not match the spec pin")


def _require_ready(seams) -> None:
    """Heartbeat, then an empty card. Both run before the server is spawned."""

    path = getattr(seams, "heartbeat", None)
    if path is None:
        from interlock.harness.runner import heartbeat_path

        path = heartbeat_path()
    wall = getattr(seams, "wall", None)
    now = wall() if callable(wall) else time.time()
    require_heartbeat(Path(path), now)
    card = getattr(seams, "card_models", None)
    models = card() if callable(card) else ["unreadable"]
    if not isinstance(models, list):
        models = ["unreadable"]
    require_card_empty(models)


def _log_text(seams, running) -> str:
    reader = getattr(seams, "log_segment", None)
    if callable(reader):
        return str(reader())
    path = getattr(running, "log_path", None)
    if path is None:
        path = getattr(seams, "log_path", None)
    if path is None:
        return ""
    return read_log_segment(Path(path), int(getattr(running, "log_offset", 0) or 0))


def _require_owned_server(spec: dict, seams, running) -> None:
    prove_owned_server(spec, segment=_log_text(seams, running), ps=seams.probe("GET", "/api/ps"))


def note_vram(read) -> dict:
    """The spec names a desktop idle band and does not give its MiB. Recorded, not gated."""

    try:
        value = None if read is None else read()
    except Exception:
        value = None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = None
    return {
        "vram_mib_at_start": value,
        "vram_note": "recorded, not gated; the spec names a desktop idle band and does not give its MiB",
    }


def parse_pairs(text: str | None) -> list[str] | None:
    """Comma-separated pair ids. None means every pair in the spec."""

    if text is None:
        return None
    names = [part.strip() for part in str(text).split(",")]
    if not names or any(not name for name in names):
        raise E4bError("--pairs names at least one pair")
    return names


def resolve_pairs(spec: dict, pairs: list[str] | None) -> tuple[list[dict], list[str] | None]:
    """The pairs this run schedules, in spec order. None records no subset."""

    listed = [pair for pair in spec.get("pairs") or [] if isinstance(pair, dict)]
    if pairs is None:
        return listed, None
    if not isinstance(pairs, (list, tuple)) or not pairs:
        raise E4bError("--pairs names at least one pair")
    known = [pair.get("id") for pair in listed]
    unknown = [name for name in pairs if name not in known]
    if unknown:
        raise E4bError(f"unknown pair: {unknown[0]}")
    if len(list(pairs)) != len(set(pairs)):
        raise E4bError("a pair is named more than once")
    chosen = [pair for pair in listed if pair.get("id") in set(pairs)]
    return chosen, [pair["id"] for pair in chosen]


def schedule_e4b(spec: dict, repeats: int) -> list[dict]:
    """Cold once, then rotated A-B-A triplets. The burn probe follows the igpu-5090 triplet."""

    pairs = list(spec["pairs"])
    if repeats < 1:
        raise E4bError("E4b needs at least one repeat")
    if not pairs:
        raise E4bError("E4b needs pairs")
    seen: list[str] = []
    items: list[dict] = []
    for pair in pairs:
        for key in ("alone", "paired"):
            arm_id = pair[key]
            if arm_id not in seen:
                seen.append(arm_id)
                items.append({"kind": "cold", "arm": arm_id})
    for index in range(repeats):
        offset = index % len(pairs)
        for pair in pairs[offset:] + pairs[:offset]:
            items.append(
                {
                    "kind": "triplet",
                    "pair": pair["id"],
                    "repeat": index,
                    "alone": pair["alone"],
                    "paired": pair["paired"],
                }
            )
            if pair["id"] == "igpu-5090":
                items.append({"kind": "probe", "probe": "igpu-after-cpu-burn", "repeat": index})
    return items


def run_burn(*, target_utility: float, duration_s: float, sample_utility, sample_performance, sleep,
    start_threads) -> dict:
    """Busy threads matched to a utility target. No GPU work is available here.

    The threads stop when this returns, and when it raises.
    """

    try:
        threads = 0
        if target_utility < 0 or duration_s < 0:
            raise E4bError("the burn target and duration cannot be negative")
        while threads < 64:
            if sample_utility() >= target_utility:
                break
            threads += 1
            start_threads(1)
        matched = sample_utility() >= target_utility
        samples = []
        elapsed = 0.0
        while elapsed + 1e-12 < duration_s:
            samples.append(
                {"t_s": elapsed, "utility": sample_utility(), "performance": sample_performance()}
            )
            step = min(1.0, duration_s - elapsed)
            sleep(step)
            elapsed += step
        return {
            "threads": threads,
            "matched": matched,
            "duration_s": duration_s,
            "utility_target": target_utility,
            "samples": samples,
            "gpu_calls": 0,
        }
    finally:
        _stop_burn(start_threads)


def pilot_summary(ranges: list[float], full_flags: list[bool], *, step: float = CLOCK_STEP_MHZ) -> dict:
    """The 95th percentile within-call range, rounded up, and whether every decode was full."""

    if not ranges:
        raise E4bError("the pilot has no clock ranges")
    if len(full_flags) != len(ranges):
        raise E4bError("the pilot pairs each clock range with a full-length flag")
    p95 = percentile(ranges, 95)
    rounded = round_up_mhz(p95, step=step)
    return {
        "calls": len(ranges),
        "p95_clock_range_mhz": p95,
        "clock_tolerance_mhz": rounded,
        "steady_band_mhz": rounded,
        "every_decode_full_length": all(full_flags),
    }


def pilot_destination(root: Path, run_id: str) -> Path:
    """Pilot rows stay out of receipts/ and out of the table."""

    _check_run_id(run_id)
    path = (root / "pilot-e4b" / f"{run_id}.json").resolve()
    banned = {"receipts", "table"}
    if any(part.lower() in banned for part in path.parts):
        raise E4bError("pilot rows never go to receipts/ or the table")
    return path


def _prompt_line(spec: dict) -> str:
    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
    target = dataset.get("target_tokens") if isinstance(dataset, dict) else None
    if isinstance(target, bool) or not isinstance(target, int):
        return "prompt: counting sequence, shortest prefix at or above the token target"
    return f"prompt: counting sequence, shortest prefix at or above {target} tokens"


def dry_text(spec: dict, blob: str, *, pilot: bool, pairs: list[str] | None = None) -> str:
    """What ``interlock run --dry-run`` prints for E4b. No device is contacted."""

    measurement = spec.get("measurement") or {}
    prompt_line = _prompt_line(spec)
    if pilot:
        lines = [
            "interlock dry-run: E4b",
            f"spec blob: {blob}",
            prompt_line,
            "pilot: 5090-alone only",
            f"measured calls: {PILOT_CALLS}",
            "num_predict: 1024",
            "warm-up: 768 tokens, at least 3 s, unmeasured, steady gate off",
            "95th percentile within-call clocks.sm range, rounded up to 7.5 MHz",
            "full-length: eval_count against num_predict",
            "rows: pilot-e4b/, not receipts/, not the table",
            "11434 and 11490 not used",
            "heartbeat not read; adapters not called",
            "devices not started (--dry-run)",
        ]
        return "\n".join(lines) + "\n"
    tolerance = measurement.get("clock_tolerance_mhz")
    band = measurement.get("steady_band_mhz")
    tolerance_line = (
        "clock_tolerance_mhz null until the pilot amendment"
        if tolerance is None
        else f"clock_tolerance_mhz: {tolerance}"
    )
    band_line = (
        "steady_band_mhz null until the pilot amendment" if band is None else f"steady_band_mhz: {band}"
    )
    lines = [
        "interlock dry-run: E4b",
        f"spec blob: {blob}",
        prompt_line,
        "own Ollama 127.0.0.1:11441 pinned to the 11434 install",
        "11434 and 11490 not used",
        "probes GET /api/version and GET /api/ps only",
        "session end is keep_alive 0 unload",
        "warm-up: NPU and iGPU one unmeasured call; 5090 768 tokens at least 3 s until steady",
        "until-steady: 10 samples at 200 ms, 2 attempts, else clock_not_steady",
        "5090 companion is sustained before the Intel warm-up: 768 tokens, at least 3 s, steady, next decode underway, "
            "else clock_not_steady",
        "refusals re-run once in place; two refusals in a pair are inconclusive",
        "Hodges-Lehmann, exact Wilcoxon at least 90%, TOST +/-0.10",
        "extension to 15 when the 10-triplet interval straddles",
        "bootstrap footnote: 50000 resamples, seed 20261012, never decides",
        "burn probe igpu-after-cpu-burn recorded, not a gate",
        tolerance_line,
        band_line,
        "pilot: interlock run experiments/e4b/spec.yaml --pilot",
        "heartbeat not read; adapters not called",
        "devices not started (--dry-run)",
    ]
    if pairs:
        lines.insert(-1, "pairs: " + ", ".join(pairs))
    return "\n".join(lines) + "\n"


def _require_amended(spec: dict) -> None:
    measurement = spec.get("measurement") or {}
    if measurement.get("clock_tolerance_mhz") is None or measurement.get("steady_band_mhz") is None:
        raise E4bError("clock_tolerance_mhz and steady_band_mhz are null until the pilot amendment")


def _check_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not run_id or any(part in run_id for part in ("/", "\\", "..")):
        raise E4bError("run_id must be a single name")


def _measurement(spec: dict) -> dict:
    measurement = spec.get("measurement")
    if not isinstance(measurement, dict):
        raise E4bError("E4b needs measurement")
    return measurement


def _clock_range(samples: list | None) -> float | None:
    if not isinstance(samples, list) or not samples or any(item is None for item in samples):
        return None
    return float(max(samples) - min(samples))


def _make_cache(root: Path, run_id: str, arm_id: str) -> str:
    path = root / "compile-cache" / run_id / arm_id
    path.mkdir(parents=True, exist_ok=True)
    return path.relative_to(root).as_posix()


def _clear_readonly(function, path, _exc) -> None:
    # OpenVINO writes compiled blobs read-only; on Windows rmtree then fails with WinError 5.
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _delete_caches(root: Path, relative: list[str]) -> list[str]:
    """Remove the run's compile caches. A cache that won't go is reported, never raised:
    a cleanup failure must not cost the receipt."""

    failed = []
    for rel in relative:
        path = root / rel
        if not path.exists():
            continue
        try:
            shutil.rmtree(path, onexc=_clear_readonly)
        except OSError as exc:
            failed.append(f"{rel}: {exc}")
    return failed


def _safe_caches(root: Path, relative: list[str]) -> list[str]:
    """Cleanup never raises. A failure is a note on the receipt, not a lost receipt."""

    try:
        return _delete_caches(root, relative)
    except Exception as exc:
        return [f"compile-cache: {exc}"]


def _note_cleanup(path: Path, failures: list[str]) -> None:
    if not failures or not path.is_file():
        return
    try:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            return
        lines[0]["compile_cache_cleanup_failed"] = failures
        _write_jsonl(path, lines)
    except (OSError, json.JSONDecodeError):
        return


def _write_series(path: Path, clocks: list, power: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, clock in enumerate(clocks):
        watts = power[index] if index < len(power) else None
        lines.append(json.dumps({"i": index, "sm_clock_mhz": clock, "power_w": watts}))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _probe(seams, method: str, path: str) -> dict:
    if method != "GET" or path not in {"/api/version", "/api/ps"}:
        raise E4bError("probes only GET /api/version and GET /api/ps")
    payload = seams.probe(method, path)
    if not isinstance(payload, dict):
        raise E4bError(f"GET {path} did not return an object")
    return payload


def warm_5090(seams, arm: dict, prompt: str, measurement: dict, *, enforce: bool) -> dict:
    """Up to two unmeasured 768-token decodes. Steady is 10 samples inside the band."""

    attempts = []
    every_s = float(measurement["steady_sample_ms"]) / 1000.0
    minimum = float(measurement["warmup_min_s"])
    predict = int(measurement["warmup_num_predict"])
    band = measurement.get("steady_band_mhz")
    count = int(measurement["steady_samples"])
    limit = int(measurement["warmup_max_attempts"])
    ok = False
    for _ in range(limit):
        _trace(seams, ("warm", arm["id"], predict))
        result = dict(
            seams.call(arm, num_predict=predict, measured=False, kind="warm", prompt=prompt)
        )
        clocks = result.get("clocks_mhz")
        if not isinstance(clocks, list):
            elapsed = float(result.get("elapsed_s") or 0)
            series = collect_series(
                sample=seams.sample_gpu,
                sleep=seams.sleep,
                every_s=every_s,
                duration_s=elapsed,
            )
            result["clocks_mhz"] = [item[0] for item in series]
            result["power_w"] = [item[1] for item in series]
            clocks = result["clocks_mhz"]
        elapsed = result.get("elapsed_s")
        if elapsed is None:
            elapsed = every_s * max(0, len(clocks) - 1)
        result["elapsed_s"] = float(elapsed)
        steady = float(elapsed) + 1e-9 >= minimum and is_steady(clocks, band=band, count=count)
        result["steady"] = steady
        attempts.append(result)
        if not enforce or steady:
            ok = True
            break
    if not enforce:
        ok = True
    return {"ok": ok, "attempts": attempts, "steady": ok}


def _clock_now(seams) -> float:
    now = getattr(seams, "now", None)
    if now is None:
        return time.perf_counter()
    return float(now())


def sustain_5090_companion(seams, arm: dict, prompt: str, measurement: dict) -> dict:
    """One steady 5090 warm, then the next decode marked underway.

    The fake path runs this on the caller. The live companion thread uses the
    same ``warm_5090`` rule and signals underway itself, so this is not called
    twice.
    """

    begun = _clock_now(seams)
    warm = warm_5090(seams, arm, prompt, measurement, enforce=True)
    finished = _clock_now(seams)
    calls = [
        {"started_s": begun, "ended_s": finished, "result": attempt, "part": "warm"}
        for attempt in warm["attempts"]
    ]
    if not warm["ok"]:
        return {"ok": False, "reason": "clock_not_steady", "calls": calls, "open": None}
    open_call = {"started_s": _clock_now(seams), "ended_s": None, "result": None, "part": "decode"}
    calls.append(open_call)
    _trace(seams, ("companion-warm-done", arm["id"]))
    _trace(seams, ("companion-decode-start", arm["id"]))
    return {"ok": True, "reason": None, "calls": calls, "open": open_call}


def _await_sustained(seams, arm: dict, prompt: str, measurement: dict) -> dict:
    waiter = getattr(seams, "await_sustained_companion", None)
    if waiter is not None:
        return waiter(arm)
    return sustain_5090_companion(seams, arm, prompt, measurement)


def _finish_companion_call(seams, arm: dict, open_call, prompt: str) -> None:
    """Fill the fake path's open decode. A live thread already holds the generate."""

    if not isinstance(open_call, dict):
        return
    if open_call.get("result") is None and getattr(seams, "await_sustained_companion", None) is None:
        open_call["result"] = dict(
            seams.call(
                arm,
                num_predict=arm.get("num_predict"),
                measured=False,
                kind="companion",
                prompt=prompt,
            )
        )
    if open_call.get("ended_s") is None:
        open_call["ended_s"] = _clock_now(seams)


def _covering_call(calls, span) -> dict | None:
    """The companion call whose time span covers the Intel measured call."""

    if not calls or span is None:
        return None
    start, end = span
    covers = []
    for index, call in enumerate(calls):
        begun = call.get("started_s")
        finished = call.get("ended_s")
        if begun is None or finished is None:
            continue
        if float(begun) <= float(start) + 1e-9 and float(finished) + 1e-9 >= float(end):
            covers.append((float(begun), index, call))
    if not covers:
        return None
    covers.sort()
    return covers[-1][2]


def _stop_kind(companion: dict) -> str:
    # 5090 stays loaded. Ending its companion is the end of the generate loop.
    if companion["device"] == "5090":
        return "end-generate"
    return "unload"


def perform_slot(
    seams,
    arm: dict,
    *,
    arms: dict,
    prompt: str,
    measurement: dict,
    role: str,
    pair: str | None,
    repeat: int,
    root: Path,
    run_id: str,
    attempts: int,
    enforce_steady: bool,
    phase: str,
) -> dict:
    """One slot. A 5090 companion is sustained before an Intel warm-up."""

    tolerance = measurement.get("clock_tolerance_mhz")
    seams.phase = phase
    companion = None
    if role == "paired":
        companion_id = arm.get("companion")
        companion = arms.get(companion_id) if companion_id else None
        if companion is None:
            raise E4bError(f"paired arm {arm['id']} needs a companion arm")
    events = 0
    final = None
    reasons = []
    attempt_time: list[dict] = []
    for _attempt in range(attempts):
        started = False
        phase_now = "start"
        cpu_start = None
        utility = None
        cpu_end = None
        warm = {"ok": False, "attempts": []}
        measured = None
        companion_calls: list = []
        open_call = None
        measure_span = None
        companion_unsteady = False
        judged = None
        try:
            if companion is not None:
                _trace(seams, ("start_companion", companion["id"]))
                seams.start_companion(companion, prompt=prompt)
                started = True
                if companion.get("device") == "5090":
                    phase_now = "companion"
                    sustained = _await_sustained(seams, companion, prompt, measurement)
                    calls = sustained.get("calls")
                    companion_calls = calls if isinstance(calls, list) else []
                    open_call = sustained.get("open")
                    if not sustained.get("ok"):
                        companion_unsteady = True
            cpu_start = seams.sample_cpu("performance")
            utility = seams.sample_cpu("utility") if arm["device"] == "5090" else None
            steady_ok = None
            if companion_unsteady:
                warm = {"ok": False, "attempts": []}
            elif arm["device"] == "5090":
                phase_now = "warm"
                warm = warm_5090(seams, arm, prompt, measurement, enforce=enforce_steady)
                steady_ok = True if not enforce_steady else warm["ok"]
                if enforce_steady and not warm["ok"]:
                    steady_ok = False
            else:
                phase_now = "warm"
                _trace(seams, ("warm", arm["id"], arm.get("num_predict")))
                seams.call(arm, num_predict=arm.get("num_predict"), measured=False, kind="warm", prompt=prompt)
                warm = {"ok": True, "attempts": []}
            if not companion_unsteady and steady_ok is not False:
                phase_now = "measure"
                measure_began = _clock_now(seams)
                _trace(seams, ("measure-start", arm["id"], role))
                measured = dict(
                    seams.call(
                        arm,
                        num_predict=arm.get("num_predict"),
                        measured=True,
                        kind="measure",
                        prompt=prompt,
                    )
                )
                measure_span = (measure_began, _clock_now(seams))
            cpu_end = seams.sample_cpu("performance")
            report = None
            not_covering = False
            if companion is not None:
                phase_now = "companion"
                kind = _stop_kind(companion)
                _trace(seams, ("stop_companion", companion["id"], kind))
                seams.stop_companion(companion, kind=kind)
                started = False
                _finish_companion_call(seams, companion, open_call, prompt)
                judged = _covering_call(companion_calls, measure_span)
                if isinstance(judged, dict) and isinstance(judged.get("result"), dict):
                    report = judged["result"]
                elif companion.get("device") == "5090" and not companion_unsteady and phase != "cold":
                    # No 5090 companion call spans the Intel measurement: refuse rather than judge the latest
                    # call, which is what block 2 did (R&D on #92).
                    not_covering = True
                elif not companion_unsteady:
                    report = seams.companion_report(companion)
            if measured is None:
                measured = {"value": None, "clocks_mhz": None, "power_w": None}
            if companion_unsteady and phase != "cold":
                reason = "clock_not_steady"
            elif not_covering:
                reason = "companion_not_covering"
            elif phase != "cold":
                reason = slot_refusal(
                    arm,
                    measured,
                    companion_arm=companion,
                    companion=report if isinstance(report, dict) else None,
                    tolerance=tolerance,
                    steady_ok=steady_ok if arm["device"] == "5090" and enforce_steady else None,
                    num_predict=arm.get("num_predict"),
                )
            else:
                reason = None
            attempt_time.extend(_phase_times(warm, measured))
            if reason:
                events += 1
                reasons.append(reason)
                final = _slot_row(
                    arm, role, pair, repeat, measured, reason, events, reasons, cpu_start, cpu_end, utility, root,
                        run_id, warm, attempt_time, judged
                )
                continue
            final = _slot_row(
                arm, role, pair, repeat, measured, None, events, reasons, cpu_start, cpu_end, utility, root, run_id,
                    warm, attempt_time, judged
            )
            break
        except Exception as exc:
            if not _is_repeat_limit(exc):
                _stop_started(seams, companion, started)
                raise
            _stop_started(seams, companion, started)
            attempt_time.extend(_phase_times(warm, None))
            attempt_time.append(_repeat_spent(exc, phase_now))
            events += 1
            reasons.append("short_decode")
            if measured is None:
                measured = {"value": None, "clocks_mhz": None, "power_w": None, "eval_count": 0}
            final = _slot_row(
                arm, role, pair, repeat, measured, "short_decode", events, reasons, cpu_start, cpu_end, utility, root,
                    run_id, warm, attempt_time
            )
            continue
    if final is None:
        raise E4bError("the slot did not run")
    return final


def _stop_started(seams, companion, started: bool) -> None:
    if not started or companion is None:
        return
    kind = _stop_kind(companion)
    _trace(seams, ("stop_companion", companion["id"], kind))
    try:
        seams.stop_companion(companion, kind=kind)
    except Exception as exc:
        if not _is_repeat_limit(exc):
            raise


def _phase_times(warm, measured) -> list[dict]:
    rows = [call_time_spent(item, kind="warm") for item in (warm or {}).get("attempts") or []]
    if isinstance(measured, dict) and measured.get("value") is not None or (
        isinstance(measured, dict) and measured.get("eval_count") is not None
    ):
        rows.append(call_time_spent(measured, kind="measure"))
    return rows


def _slot_row(arm, role, pair, repeat, measured, reason, events, reasons, cpu_start, cpu_end, utility, root, run_id,
    warm, attempt_time=None, companion_call=None) -> dict:
    clocks = measured.get("clocks_mhz") if isinstance(measured.get("clocks_mhz"), list) else []
    power = measured.get("power_w") if isinstance(measured.get("power_w"), list) else []
    series_rel = None
    if arm["device"] == "5090" and clocks:
        series = root / "receipts" / f"{run_id}-series" / f"{pair or 'cold'}-{repeat}-{role}-{arm['id']}.jsonl"
        _write_series(series, clocks, power)
        series_rel = series.relative_to(root).as_posix()
    row = {
        "arm": arm["id"],
        "role": role,
        "pair": pair,
        "repeat": repeat,
        "value": measured.get("value"),
        "refusal": reason,
        "refusal_events": events,
        "attempt_reasons": list(reasons),
        "load_duration_s": measured.get("load_duration_s"),
        "eval_count": measured.get("eval_count"),
        "truncated_tokens": measured.get("truncated_tokens"),
        "clock_range_mhz": _clock_range(clocks),
        "cpu_perf_start": cpu_start,
        "cpu_perf_end": cpu_end,
        "cpu_utility": utility,
        "elapsed_s": measured.get("elapsed_s"),
        "series": series_rel,
        "warm_attempts": len(warm.get("attempts") or []),
        "time_spent": list(attempt_time or []),
        "analysis": role != "cold",
    }
    if isinstance(companion_call, dict):
        result = companion_call.get("result") if isinstance(companion_call.get("result"), dict) else {}
        clocks = result.get("clocks_mhz")
        row["companion_clock_range_mhz"] = _clock_range(clocks if isinstance(clocks, list) else None)
        row["companion_call_started_s"] = companion_call.get("started_s")
        row["companion_call_ended_s"] = companion_call.get("ended_s")
        row["companion_call_part"] = companion_call.get("part")
    return row


def _triplet(seams, item, arms, prompt, measurement, root, run_id, *, attempts: int, enforce_steady: bool) -> dict:
    roles = (
        ("alone-before", item["alone"]),
        ("paired", item["paired"]),
        ("alone-after", item["alone"]),
    )
    slots = []
    added = 0
    for role, arm_id in roles:
        if added >= 2:
            slots.append({"arm": arm_id, "role": role, "pair": item["pair"], "repeat": item["repeat"], "skipped": True,
                "value": None, "refusal_events": 0})
            continue
        slot = perform_slot(
            seams,
            arms[arm_id],
            arms=arms,
            prompt=prompt,
            measurement=measurement,
            role=role,
            pair=item["pair"],
            repeat=item["repeat"],
            root=root,
            run_id=run_id,
            attempts=attempts,
            enforce_steady=enforce_steady,
            phase="repeat",
        )
        added += slot["refusal_events"]
        slots.append(slot)
        if added >= 2:
            continue
    values = _triplet_values(slots)
    delta = None
    if values is not None and added < 2:
        delta = effect(values[0], values[2], values[1])
    return {"slots": slots, "events": added, "effect": delta}


def _triplet_values(slots: list[dict]) -> tuple[float, float, float] | None:
    by_role = {slot.get("role"): slot for slot in slots}
    needed = ("alone-before", "paired", "alone-after")
    picked = []
    for role in needed:
        slot = by_role.get(role)
        if slot is None or slot.get("skipped") or slot.get("refusal") or slot.get("value") is None:
            return None
        picked.append(slot["value"])
    return picked[0], picked[1], picked[2]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _run_probe(seams, probe_arm, prompt, measurement, root, run_id, repeat, footprints, durations) -> dict:
    target = _mean(footprints)
    duration = durations[-1] if durations else 0.0
    if target is None:
        burn = {
            "threads": 0,
            "matched": False,
            "duration_s": duration,
            "utility_target": None,
            "samples": [],
            "gpu_calls": 0,
        }
    else:
        burn = run_burn(
            target_utility=target,
            duration_s=duration,
            sample_utility=lambda: seams.sample_cpu("utility"),
            sample_performance=lambda: seams.sample_cpu("performance"),
            sleep=seams.sleep,
            start_threads=seams.start_threads,
        )
    slot = perform_slot(
        seams,
        probe_arm,
        arms={probe_arm["id"]: probe_arm},
        prompt=prompt,
        measurement=measurement,
        role="probe",
        pair=None,
        repeat=repeat,
        root=root,
        run_id=run_id,
        attempts=1,
        enforce_steady=True,
        phase="probe",
    )
    return {
        "kind": "e4b-probe",
        "id": probe_arm["id"],
        "repeat": repeat,
        "gate": False,
        "threads": burn["threads"],
        "matched": burn["matched"],
        "duration_s": burn["duration_s"],
        "utility_target": burn["utility_target"],
        "samples": burn["samples"],
        "gpu_calls": 0,
        "igpu_latency_ms": slot.get("value"),
        "slot": slot,
    }


def _analyze(effects: list[float], *, band: float, resamples: int, allow_extension: bool) -> dict:
    if not effects:
        return {
            "d_i": [],
            "hodges_lehmann": None,
            "low": None,
            "high": None,
            "achieved": None,
            "verdict": "inconclusive",
            "side": None,
            "straddles": False,
            "extension_used": False,
            "bootstrap": None,
        }
    decision = decide(effects, band=band, allow_extension=allow_extension)
    decision["bootstrap"] = bootstrap_median(effects, resamples=resamples, seed=20261012)
    return decision


def _extend_pair(seams, pair, arms, prompt, measurement, root, run_id, *, start_repeat: int, extra: int) -> list[dict]:
    found = []
    for offset in range(extra):
        item = {
            "kind": "triplet",
            "pair": pair["id"],
            "repeat": start_repeat + offset,
            "alone": pair["alone"],
            "paired": pair["paired"],
        }
        found.append(
            _triplet(seams, item, arms, prompt, measurement, root, run_id, attempts=2, enforce_steady=True)
        )
    return found


def _partial_pairs(spec: dict, per_pair: dict) -> list[dict]:
    """Slots that finished before a stop. No verdict: the header carries aborted."""

    rows = []
    for pair in spec["pairs"]:
        state = per_pair[pair["id"]]
        if not state["slots"] and not state["effects"]:
            continue
        rows.append(
            {
                "kind": "e4b-pair",
                "pair": pair["id"],
                "slots": state["slots"],
                "d_i": list(state["effects"]),
                "refusal_events": state["events"],
            }
        )
    return rows


def execute_block(spec: dict, seams, *, root: Path, run_id: str, pairs: list[str] | None = None) -> dict:
    """The measured block. Refusals, the estimate, and the burn probe land on the receipt."""

    try:
        return _execute_block_loop(spec, seams, root=root, run_id=run_id, pairs=pairs)
    except BaseException as exc:
        # KeyboardInterrupt is an abort too. The rows finished so far travel with it.
        snap = getattr(_execute_block_loop, "last_partial", None)
        _execute_block_loop.last_partial = None
        if isinstance(snap, dict):
            exc.e4b_partial = snap
        raise


def _execute_block_loop(spec: dict, seams, *, root: Path, run_id: str, pairs: list[str] | None = None) -> dict:
    chosen, _recorded = resolve_pairs(spec, pairs)
    spec = dict(spec)
    spec["pairs"] = chosen
    measurement = _measurement(spec)
    prompt = prepare_prompt(spec, seams.count_tokens)
    arms = {arm["id"]: arm for arm in spec["arms"]}
    repeats = int(spec["repeats"])
    band = float(spec["band"])
    resamples = int(getattr(seams, "resamples", 50000))
    items = schedule_e4b(spec, repeats)
    per_pair: dict[str, dict] = {
        pair["id"]: {"effects": [], "events": 0, "slots": [], "triplets": []} for pair in spec["pairs"]
    }
    cold = []
    probes = []
    footprints: list[float] = []
    durations: list[float] = []
    probe_arm = _probe_arm(spec)
    for item in items:
        _execute_block_loop.last_partial = {
            "prompt": prompt,
            "cold": list(cold),
            "pairs": _partial_pairs(spec, per_pair),
            "probes": list(probes),
        }
        if item["kind"] == "cold":
            cold.append(
                perform_slot(
                    seams,
                    arms[item["arm"]],
                    arms=arms,
                    prompt=prompt,
                    measurement=measurement,
                    role="cold",
                    pair=None,
                    repeat=0,
                    root=root,
                    run_id=run_id,
                    attempts=1,
                    enforce_steady=True,
                    phase="cold",
                )
            )
            continue
        if item["kind"] == "probe":
            if probe_arm is None:
                raise E4bError("E4b names the igpu-after-cpu-burn probe")
            probes.append(
                _run_probe(seams, probe_arm, prompt, measurement, root, run_id, item["repeat"], footprints, durations)
            )
            continue
        state = per_pair[item["pair"]]
        if state["events"] >= 2:
            continue
        done = _triplet(seams, item, arms, prompt, measurement, root, run_id, attempts=2, enforce_steady=True)
        state["events"] += done["events"]
        state["slots"].extend(done["slots"])
        state["triplets"].append(done)
        if state["events"] < 2 and done["effect"] is not None:
            state["effects"].append(done["effect"])
        _collect_5090(done["slots"], footprints, durations)
    pairs_out = []
    extension_limit = int(spec.get("extension_repeats") or repeats)
    extra = extension_limit - repeats
    for pair in spec["pairs"]:
        state = per_pair[pair["id"]]
        if state["events"] >= 2:
            record = _analyze([], band=band, resamples=resamples, allow_extension=False)
            record["d_i"] = list(state["effects"])
            record["verdict"] = "inconclusive"
            record["refusal_events"] = state["events"]
        else:
            record = _analyze(state["effects"], band=band, resamples=resamples,
                allow_extension=len(state["effects"]) == 10)
            if record["verdict"] == "extend" and extra > 0:
                more = _extend_pair(
                    seams, pair, arms, prompt, measurement, root, run_id, start_repeat=repeats, extra=extra
                )
                added_events = 0
                for done in more:
                    added_events += done["events"]
                    state["slots"].extend(done["slots"])
                    if done["effect"] is not None and added_events < 2 and state["events"] + added_events < 2:
                        state["effects"].append(done["effect"])
                state["events"] += added_events
                record = _analyze(state["effects"], band=band, resamples=resamples, allow_extension=False)
                record["extension_used"] = True
                if state["events"] >= 2:
                    record["verdict"] = "inconclusive"
            else:
                record["extension_used"] = False
            record["refusal_events"] = state["events"]
        record["kind"] = "e4b-pair"
        record["pair"] = pair["id"]
        record["slots"] = state["slots"]
        pairs_out.append(record)
    _execute_block_loop.last_partial = None
    return {"prompt": prompt, "cold": cold, "pairs": pairs_out, "probes": probes}


def _collect_5090(slots: list[dict], footprints: list[float], durations: list[float]) -> None:
    for slot in slots:
        if slot.get("skipped") or slot.get("role") == "cold":
            continue
        if slot.get("cpu_utility") is not None and slot.get("arm", "").startswith("5090"):
            footprints.append(float(slot["cpu_utility"]))
        if slot.get("arm", "").startswith("5090") and slot.get("elapsed_s") is not None and slot.get("role") != "cold":
            durations.append(float(slot["elapsed_s"]))


def _probe_arm(spec: dict) -> dict | None:
    for probe in spec.get("probes") or []:
        if isinstance(probe, dict) and probe.get("id") == "igpu-after-cpu-burn":
            return probe
    return None


def _server_config_line(segment: str) -> str | None:
    last = None
    for line in segment.splitlines():
        if 'msg="server config"' in line:
            last = line
    return last


def _sample_note(seams) -> dict:
    """The probe that answered, and the mean gap between its samples."""

    record = getattr(seams, "sample_record", None)
    intervals = [] if record is None else list(getattr(record, "intervals", []) or [])
    return {
        "gpu_probe": None if record is None else getattr(record, "probe", None),
        "sample_interval_s": _mean(intervals),
    }


def _clocks_missing(clocks) -> bool:
    return not isinstance(clocks, list) or not clocks or any(item is None for item in clocks)


def _git_at_start(seams, repo: Path, git: dict | None) -> dict:
    """The sha and clean flag from before this run writes. A passed dict is not read again."""

    if git is not None:
        return git
    read = getattr(seams, "git", None)
    if read:
        return read(repo)
    return {"git_sha": None, "clean_tree": None}


def _header(spec, seams, repo, running, caches, prompt, *, aborted: bool, git: dict | None = None,
    pairs: list[str] | None = None) -> dict:
    from interlock.ollama_supervisor import server_config_env

    recorded = git if git is not None else {"git_sha": None, "clean_tree": None}
    vram = note_vram(getattr(seams, "vram", None))
    segment = _log_text(seams, running)
    config = server_config_env(segment)
    facts = getattr(seams, "load_facts", None)
    if not isinstance(facts, dict):
        facts = _load_facts(segment)
    usable = isinstance(facts, dict) and not facts.get("reason")
    header = {
        "kind": "e4b-header",
        "id": spec["id"],
        "git_sha": recorded.get("git_sha"),
        "clean_tree": recorded.get("clean_tree"),
        "argv": list(getattr(running, "argv", []) or []),
        "environment": receipt_environment(dict(getattr(running, "environment", {}) or {})),
        "version_start": getattr(running, "version_start", None),
        "version_stop": getattr(running, "version_stop", None),
        "caches": caches,
        "prompt_sha256": _sha256(prompt),
        "aborted": aborted,
        "server_config_line": _server_config_line(segment),
        "flash_attention": _flash_flag(facts.get("flash_attention")) if usable else None,
        "kv_cache_type": facts.get("kv_cache_type") if usable else None,
        "config_flash_attention": config.get("OLLAMA_FLASH_ATTENTION"),
    }
    header.update(vram)
    header.update(_sample_note(seams))
    if pairs is not None:
        header["pairs"] = list(pairs)
    return header


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _park(
    root: Path,
    run_id: str,
    *,
    sha: str | None,
    started: str,
    exc: BaseException | None,
    trace: str | None,
    quarantine: Path | None,
) -> None:
    """Move the receipt and the partial outputs aside. A failure here does not replace the stop."""

    try:
        move_aside(
            root,
            run_id,
            sha=sha,
            started=started,
            ended=utc_now(),
            exc_type=type(exc).__name__ if exc is not None else None,
            message=str(exc) if exc is not None else "",
            traceback_text=trace,
            include_receipt=True,
            setting=quarantine,
        )
    except Exception:
        return


def run_block(
    spec: dict,
    seams,
    *,
    root: Path,
    run_id: str,
    repo: Path,
    git: dict | None = None,
    quarantine: Path | None = None,
    pairs: list[str] | None = None,
) -> list[dict]:
    """One E4b block. The server's process tree stops on the way out, including on abort."""

    _check_run_id(run_id)
    _require_amended(spec)
    recorded_pairs = None
    if pairs is not None:
        _chosen, recorded_pairs = resolve_pairs(spec, pairs)
    require_free(root, run_id, quarantine)
    refuse_strays(root)
    receipt = root / "receipts" / f"{run_id}.jsonl"
    # Before caches, the server, and the receipt. The header keeps this read.
    recorded = _git_at_start(seams, repo, git)
    refuse_dirty_worktree(recorded)
    started_at = utc_now()
    arms = {arm["id"]: arm for arm in spec["arms"]}
    caches = [_make_cache(root, run_id, arm_id) for arm_id in arms]
    rows: list[dict] = []
    started = False
    aborted = False
    running = None
    prompt = ""
    error: BaseException | None = None
    trace: str | None = None
    cleanup_errors: list[str] = []
    _require_ready(seams)
    try:
        with seams.server as running:
            started = True
            _probe(seams, "GET", "/api/version")
            _probe(seams, "GET", "/api/ps")
            body_error: BaseException | None = None
            try:
                seams.load_resident()
                _require_owned_server(spec, seams, running)
                seams.load_facts = _require_load_lines(seams, running)
                body = execute_block(spec, seams, root=root, run_id=run_id, pairs=pairs)
                rows = _rows_of(body)
                prompt = body["prompt"]
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                if body_error is None:
                    try:
                        seams.unload_resident()
                    finally:
                        _probe(seams, "GET", "/api/version")
                else:
                    # A cleanup error must not replace the abort or drop its partial rows.
                    try:
                        seams.unload_resident()
                    except Exception as exc:
                        cleanup_errors.append(f"unload_resident: {exc}")
                    try:
                        _probe(seams, "GET", "/api/version")
                    except Exception as exc:
                        cleanup_errors.append(f"GET /api/version: {exc}")
    except BaseException as exc:
        aborted = True
        error = exc
        trace = traceback.format_exc()
        partial = getattr(exc, "e4b_partial", None)
        if isinstance(partial, dict):
            rows = _rows_of(partial)
            prompt = partial.get("prompt") or prompt
    finally:
        # The receipt lands before cleanup. Cleanup cannot raise past it.
        if started and running is not None:
            header = _header(spec, seams, repo, running, caches, prompt, aborted=aborted, git=recorded,
                pairs=recorded_pairs)
            header["run_id"] = run_id
            if cleanup_errors:
                header["cleanup_errors"] = cleanup_errors
            try:
                _write_jsonl(receipt, [header, *rows])
            except OSError:
                if error is None:
                    raise
            else:
                _note_cleanup(receipt, _safe_caches(root, caches))
                if aborted:
                    sha = recorded.get("git_sha") if isinstance(recorded, dict) else None
                    _park(root, run_id, sha=sha, started=started_at, exc=error, trace=trace, quarantine=quarantine)
        else:
            _safe_caches(root, caches)
    if error is not None:
        raise error
    return rows


def _rows_of(body: dict) -> list[dict]:
    rows = []
    for slot in body["cold"]:
        slot = dict(slot)
        slot["kind"] = "e4b-cold"
        slot["analysis"] = False
        rows.append(slot)
    rows.extend(body["pairs"])
    rows.extend(body["probes"])
    return rows


def run_pilot(
    spec: dict,
    seams,
    *,
    root: Path,
    run_id: str,
    repo: Path,
    calls: int = PILOT_CALLS,
    git: dict | None = None,
    quarantine: Path | None = None,
) -> dict:
    """5090-alone only. Rows go to pilot-e4b/ and never into receipts/ or the table."""

    _check_run_id(run_id)
    require_free(root, run_id, quarantine)
    refuse_strays(root)
    destination = pilot_destination(root, run_id)
    arm = next((item for item in spec["arms"] if item["id"] == "5090-alone"), None)
    if arm is None:
        raise E4bError("the pilot measures 5090-alone")
    # Before the server and before pilot-e4b/ is written. The document keeps this read.
    recorded = _git_at_start(seams, repo, git)
    refuse_dirty_worktree(recorded)
    measurement = _measurement(spec)
    ranges = []
    full = []
    warm_short = 0
    missing = 0
    call_intervals: list[float] = []
    spent_rows: list[dict] = []
    started_at = utc_now()
    stopped: list = []
    cleanup_errors: list[str] = []
    _require_ready(seams)
    with seams.server as running:
        _probe(seams, "GET", "/api/version")
        _probe(seams, "GET", "/api/ps")
        try:
            seams.load_resident()
            _require_owned_server(spec, seams, running)
            seams.load_facts = _require_load_lines(seams, running)
            # The prompt is counted on the run-owned server, so only once it is up and proven.
            prompt = prepare_prompt(spec, seams.count_tokens)
            for index in range(calls):
                try:
                    warm = warm_5090(seams, arm, prompt, measurement, enforce=False)
                except Exception as exc:
                    if not _is_repeat_limit(exc):
                        raise
                    missing += 1
                    spent_rows.append({"call": index, "missing": True, "phases": [_repeat_spent(exc, "warm")]})
                    continue
                if not warm["attempts"] or float(warm["attempts"][-1].get("elapsed_s") or 0) + 1e-9 < float(measurement[
                    "warmup_min_s"]):
                    warm_short += 1
                try:
                    measured = dict(
                        seams.call(arm, num_predict=arm.get("num_predict"), measured=True, kind="measure",
                            prompt=prompt)
                    )
                except Exception as exc:
                    if not _is_repeat_limit(exc):
                        raise
                    missing += 1
                    phases = [call_time_spent(item, kind="warm") for item in warm.get("attempts") or []]
                    phases.append(_repeat_spent(exc, "measure"))
                    spent_rows.append({"call": index, "missing": True, "phases": phases})
                    continue
                interval = measured.get("sample_interval_s")
                if isinstance(interval, (int, float)) and not isinstance(interval, bool):
                    call_intervals.append(float(interval))
                phases = [call_time_spent(item, kind="warm") for item in warm.get("attempts") or []]
                phases.append(call_time_spent(measured, kind="measure"))
                clocks = measured.get("clocks_mhz")
                if _clocks_missing(clocks):
                    missing += 1
                    spent_rows.append({"call": index, "missing": True, "reason": "missing clock", "phases": phases})
                    continue
                spent_rows.append({"call": index, "missing": False, "phases": phases})
                ranges.append(float(max(clocks) - min(clocks)))
                count = measured.get("eval_count")
                predict = arm.get("num_predict")
                full.append(isinstance(count, int) and not isinstance(count, bool) and count >= predict)
        except BaseException as exc:
            stopped.append(exc)
            stopped.append(traceback.format_exc())
        finally:
            if stopped:
                try:
                    seams.unload_resident()
                except Exception as exc:
                    cleanup_errors.append(f"unload_resident: {exc}")
            else:
                seams.unload_resident()
    if stopped:
        parked = {
            "kind": "e4b-pilot",
            "run_id": run_id,
            "aborted": True,
            "calls": calls,
            "good_calls": len(ranges),
            "missing_calls": missing,
            "ranges_mhz": ranges,
            "time_spent": spent_rows,
            "git_sha": recorded.get("git_sha"),
            "clean_tree": recorded.get("clean_tree"),
        }
        if cleanup_errors:
            parked["cleanup_errors"] = cleanup_errors
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(parked, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _safe_caches(root, [f"compile-cache/{run_id}"])
        sha = recorded.get("git_sha") if isinstance(recorded, dict) else None
        _park(root, run_id, sha=sha, started=started_at, exc=stopped[0], trace=stopped[1], quarantine=quarantine)
        raise stopped[0]
    needed = PILOT_MIN_GOOD if calls >= PILOT_MIN_GOOD else calls
    if len(ranges) < needed:
        raise E4bError(
            f"the pilot has {len(ranges)} good calls and {missing} missing; "
            f"it needs at least {needed}. {PILOT_CLOCK_RULE}"
        )
    summary = pilot_summary(ranges, full, step=float(measurement.get("clock_step_mhz") or CLOCK_STEP_MHZ))
    document = {
        "kind": "e4b-pilot",
        "run_id": run_id,
        "aborted": False,
        "calls": calls,
        "num_predict": arm.get("num_predict"),
        "warmup_num_predict": measurement.get("warmup_num_predict"),
        "warmups_under_min_s": warm_short,
        "ranges_mhz": ranges,
        "git_sha": recorded.get("git_sha"),
        "clean_tree": recorded.get("clean_tree"),
        "argv": list(getattr(running, "argv", []) or []),
        "environment": receipt_environment(dict(getattr(running, "environment", {}) or {})),
        "version_start": getattr(running, "version_start", None),
        "version_stop": getattr(running, "version_stop", None),
    }
    document.update(summary)
    document["calls"] = calls
    document["good_calls"] = len(ranges)
    document["missing_calls"] = missing
    document["time_spent"] = spent_rows
    document["pilot_clock_rule"] = PILOT_CLOCK_RULE
    note = _sample_note(seams)
    if note["sample_interval_s"] is None and call_intervals:
        note["sample_interval_s"] = _mean(call_intervals)
    document.update(note)
    document.update(note_vram(getattr(seams, "vram", None)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _safe_caches(root, [f"compile-cache/{run_id}"])
    return document


def run_from_cli(
    spec: dict,
    *,
    spec_path: Path,
    root: Path,
    repo: Path,
    grant: str | None,
    npu_session: str | None,
    run_id: str | None,
    pilot: bool,
    seams,
    notes: str = "",
    quarantine: Path | None = None,
    pairs: list[str] | None = None,
) -> list | dict:
    """Flag checks, then the pilot or the block. Null tolerances refuse before any device."""

    from interlock.harness.lock import require_unmodified

    require_unmodified(repo, spec_path)
    if pairs is not None:
        if pilot:
            raise E4bError("--pairs is for the block, not the pilot")
        resolve_pairs(spec, pairs)
    require_grant(grant, uses_5090=True)
    if not pilot:
        require_npu_session(npu_session, device="NPU")
        require_npu_session(npu_session, device="iGPU")
        _require_amended(spec)
    # Stray series and a taken name are refused here, before a device starts.
    name = resolve_run_id(
        root,
        base="e4b-pilot" if pilot else "e4b-block",
        explicit=run_id,
        setting=quarantine,
    )
    if seams is None:
        seams = build_live_seams(root, name, spec, pilot=pilot)
    # One read, before any device. The pilot document and the block receipt keep it.
    recorded = seams.git(repo) if getattr(seams, "git", None) else None
    if pilot:
        return run_pilot(spec, seams, root=root, run_id=name, repo=repo, git=recorded, quarantine=quarantine)
    rows = run_block(spec, seams, root=root, run_id=name, repo=repo, git=recorded, quarantine=quarantine, pairs=pairs)
    if notes:
        receipt = root / "receipts" / f"{name}.jsonl"
        if receipt.is_file():
            lines = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line]
            if lines:
                lines[0]["notes"] = notes
                _write_jsonl(receipt, lines)
    return rows


def _refuse_base(base: str) -> None:
    from urllib.parse import urlsplit

    port = urlsplit(base).port
    if port in SHARED_PORTS:
        raise E4bError("11434 and 11490 are never used")
    if port != PORT:
        raise E4bError(f"the run-owned server is {HOST}:{PORT}")


def owned_generate(transport, base: str, arm: dict, prompt: str, num_predict: int | None) -> dict:
    """One generate on the run-owned server. keep_alive -1 leaves qwen3 loaded."""

    _refuse_base(base)
    body = {
        "model": arm["model"],
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "keep_alive": -1,
        "think": False,
        "options": {},
    }
    if num_predict is not None:
        body["options"]["num_predict"] = num_predict
    payload = transport("POST", f"{base}/api/generate", body)
    if not isinstance(payload, dict):
        raise E4bError("Ollama returned a non-object")
    return payload


def unload_owned(transport, base: str, model: str) -> dict:
    """The only keep_alive 0 for qwen3. There is no companion-stop generate."""

    _refuse_base(base)
    body = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    payload = transport("POST", f"{base}/api/generate", body)
    if not isinstance(payload, dict):
        raise E4bError("Ollama returned a non-object")
    return payload


def count_owned(transport, base: str, model: str, text: str) -> int:
    """Token count on the run-owned server. num_predict is 1, never 0."""

    _refuse_base(base)
    body = {
        "model": model,
        "prompt": text,
        "stream": False,
        "raw": True,
        "keep_alive": -1,
        "options": {"num_predict": 1},
    }
    payload = transport("POST", f"{base}/api/generate", body)
    count = payload.get("prompt_eval_count") if isinstance(payload, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise E4bError("Ollama did not report prompt_eval_count")
    return count


def normalize_ollama(payload: dict) -> dict:
    """Server fields the refusal rules read. Durations in the fixture are nanoseconds."""

    load_ns = payload.get("load_duration")
    eval_count = payload.get("eval_count")
    eval_ns = payload.get("eval_duration")
    load_s = None
    if isinstance(load_ns, (int, float)) and not isinstance(load_ns, bool):
        load_s = float(load_ns) / 1e9
    value = None
    elapsed = None
    if isinstance(eval_ns, (int, float)) and not isinstance(eval_ns, bool) and eval_ns > 0:
        elapsed = float(eval_ns) / 1e9
        if isinstance(eval_count, int) and not isinstance(eval_count, bool):
            value = eval_count / elapsed
    normalized = {
        "value": value,
        "load_duration_s": load_s,
        "eval_count": eval_count if isinstance(eval_count, int) and not isinstance(eval_count, bool) else None,
        "elapsed_s": elapsed,
        "truncated_tokens": None,
        "prompt_eval_count": payload.get("prompt_eval_count") if isinstance(payload.get("prompt_eval_count"),
            int) and not isinstance(payload.get("prompt_eval_count"), bool) else None,
    }
    normalized["time_spent"] = call_time_spent(payload, kind="generate")
    return normalized


def threaded_sample(invoke, sample, every_s: float, *, intervals: list | None = None) -> dict:
    """Sample while ``invoke`` runs. Tests that need a deterministic series skip this.

    ``intervals`` receives the measured gap between sample starts. A slow probe
    stretches the requested step, and that stretch is what the receipt records.
    """

    series: list[tuple] = []
    gaps: list[float] = []
    stop = threading.Event()

    def loop() -> None:
        previous = None
        while not stop.is_set():
            now = time.perf_counter()
            if previous is not None:
                gaps.append(now - previous)
            previous = now
            series.append(tuple(sample()))
            if stop.wait(every_s):
                break

    thread = threading.Thread(target=loop, name="interlock-e4b-sample", daemon=True)
    thread.start()
    try:
        result = dict(invoke())
    finally:
        stop.set()
        thread.join(timeout=every_s + 1)
    if intervals is not None:
        intervals.extend(gaps)
    if "clocks_mhz" not in result:
        result["clocks_mhz"] = [item[0] for item in series]
        result["power_w"] = [item[1] for item in series]
    if gaps:
        result["sample_interval_s"] = sum(gaps) / len(gaps)
    return result


def generate_loop(arm: dict, post, stop: threading.Event) -> dict:
    """Back-to-back generates until ``stop`` is set. The last payload is the companion report."""

    last: dict = {}
    while not stop.is_set():
        payload = post(arm)
        if isinstance(payload, dict):
            last = payload
        if stop.is_set():
            break
    return last


def _run_5090_companion(seams, arm: dict, prompt: str, measurement: dict, stop: threading.Event, box: dict,
    record_latest) -> None:
    """Warm under the 5090 rule, then keep generating. Underway fires as the next decode is entered."""

    begun = time.perf_counter()
    warm = warm_5090(seams, arm, prompt, measurement, enforce=True)
    finished = time.perf_counter()
    for attempt in warm["attempts"]:
        if isinstance(attempt, dict):
            record_latest(attempt)
        box["calls"].append(
            {"started_s": begun, "ended_s": finished, "result": attempt, "part": "warm"}
        )
    if not warm["ok"]:
        box["ok"] = False
        box["reason"] = "clock_not_steady"
        return
    open_call = {"started_s": None, "ended_s": None, "result": None, "part": "decode"}
    box["calls"].append(open_call)
    box["open"] = open_call
    first = True

    def post(item):
        nonlocal first
        started = time.perf_counter()
        if first:
            first = False
            open_call["started_s"] = started
            box["ok"] = True
            _trace(seams, ("companion-warm-done", arm["id"]))
            _trace(seams, ("companion-decode-start", arm["id"]))
            box["event"].set()
        payload = seams.call(
            item,
            num_predict=item.get("num_predict"),
            measured=False,
            kind="companion",
            prompt=prompt,
        )
        ended = time.perf_counter()
        if isinstance(payload, dict):
            record_latest(payload)
        if open_call.get("result") is None:
            open_call["ended_s"] = ended
            open_call["result"] = payload
        else:
            box["calls"].append(
                {"started_s": started, "ended_s": ended, "result": payload, "part": "decode"}
            )
        return payload

    generate_loop(arm, post, stop)


def spawn_ollama(argv: list[str], environment: dict, log_path: Path, *, popen=subprocess.Popen,
    system: str | None = None):
    """The harness's own unelevated child. Logging goes to ``log_path``."""

    system = os.name if system is None else system
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    kwargs = {"env": environment, "stdout": handle, "stderr": subprocess.STDOUT}
    if system == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    return popen(argv, **kwargs)


def _trace(seams, item) -> None:
    trace = getattr(seams, "trace", None)
    if trace is not None:
        trace.append(item)


def _http_timeout(url: str) -> float:
    """Probes are short. A generate is allowed a few minutes, then it is a refusal."""

    from urllib.parse import urlsplit

    if urlsplit(url).path == "/api/generate":
        return GENERATE_TIMEOUT_S
    return PROBE_TIMEOUT_S


def urllib_exchange(method: str, url: str, body: dict | None = None) -> dict:
    """The live HTTP call. Tests pass their own transport and never reach this."""

    import urllib.error
    import urllib.request
    from urllib.parse import urlsplit

    timeout = _http_timeout(url)
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(  # noqa: S310  # the live caller builds http://127.0.0.1
        url, data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310  # loopback http only
            payload = json.loads(response.read().decode("utf-8"))
    except TimeoutError as exc:
        path = urlsplit(url).path or url
        raise E4bError(f"REFUSED: {method} {path} timed out after {timeout:g} s") from exc
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
        path = urlsplit(url).path or url
        if exc.code == 500 and "token repeat limit" in text.lower():
            raise RepeatLimit(
                f"REFUSED: {method} {path} token repeat limit",
                call_time_spent(_json_object(text), kind="generate", reason="token repeat limit"),
            ) from exc
        raise
    if not isinstance(payload, dict):
        raise E4bError("Ollama returned a non-object")
    return payload


urllib_exchange.probe_timeout_s = PROBE_TIMEOUT_S
urllib_exchange.generate_timeout_s = GENERATE_TIMEOUT_S


def owned_probe(base: str, path: str, opener) -> dict:
    """GET only, and only on the run-owned port."""

    _refuse_base(base)
    if path not in {"/api/version", "/api/ps"}:
        raise E4bError("probes only GET /api/version and GET /api/ps")
    return get_json(f"{base}{path}", opener)


class IntelLane:
    """OpenVINO measured calls and companion sessions. The adapter is built on first use."""

    def __init__(self, root: Path, run_id: str, adapter=None):
        self.root = root
        self.run_id = run_id
        self._adapter = adapter
        self._requests: dict[str, object] = {}
        self._last_box: dict = {}

    def adapter(self):
        if self._adapter is None:
            from interlock.adapters.openvino import OpenvinoAdapter

            self._adapter = OpenvinoAdapter()
        return self._adapter

    def request(self, arm: dict, prompt: str):
        from interlock.adapters.base import StepRequest

        cache = self.root / "compile-cache" / self.run_id / arm["id"]
        cache.mkdir(parents=True, exist_ok=True)
        settings = {
            "prompt": prompt,
            "static_shape": arm.get("static_shape"),
            "batch": arm.get("batch") or 1,
            "compile_cache": "fresh",
            "compile_cache_dir": str(cache),
            "thinking_level": arm.get("thinking_level", "off"),
        }
        request = StepRequest(
            step_id=arm["id"],
            task=arm["task"],
            model=arm["model"],
            device=arm["device"],
            engine="openvino",
            settings=settings,
        )
        self._requests[arm["id"]] = request
        return request

    def call(self, arm, *, num_predict, measured, kind, prompt):
        del num_predict, measured, kind
        result = self.adapter().run(self.request(arm, prompt))
        if not getattr(result, "ok", False):
            raise E4bError(getattr(result, "error", None) or "the Intel call failed")
        resolved = result.resolved if isinstance(getattr(result, "resolved", None), dict) else {}
        latency = getattr(result, "latency_ms", None)
        return {
            "value": latency,
            "truncated_tokens": resolved.get("truncated_tokens"),
            "load_duration_s": None,
            "eval_count": None,
            "elapsed_s": None if latency is None else float(latency) / 1000,
        }

    def start(self, arm, *, prompt: str):
        self.adapter().start_session(self.request(arm, prompt))

    def stop(self, arm):
        session = getattr(self.adapter(), "_session", None)
        if isinstance(session, dict) and isinstance(session.get("box"), dict):
            self._last_box = dict(session["box"])
        request = self._requests.get(arm["id"]) or self.request(arm, "")
        self.adapter().stop_session(request)

    def unload(self, arm):
        self.adapter().unload(arm["model"], arm["device"])

    def report(self, arm) -> dict:
        session = getattr(self.adapter(), "_session", None)
        box = session.get("box") if isinstance(session, dict) else None
        if not isinstance(box, dict):
            box = self._last_box
        call_s = box.get("call_s")
        return {
            "value": None if call_s is None else float(call_s) * 1000,
            "truncated_tokens": box.get("truncated_tokens"),
            "load_duration_s": None,
            "eval_count": None,
            "elapsed_s": call_s,
        }


def build_live_seams(root: Path, run_id: str, spec: dict, *, pilot: bool, run=None, popen=None, transport=None,
    opener=None, sample=None, cpu=None, intel=None, sampler=threaded_sample, vram=None, git_run=None, bind=None,
        card_mib=None):
    """Wire the run-owned server. Every external call is injectable; the defaults are the live ones."""

    from interlock.harness.ollama_serve import Server

    runner = run or capture_run
    pin = resolve_pin(runner)
    log_root = root / ("pilot-e4b" if pilot else "logs")
    post = transport or urllib_exchange
    opener = opener or (lambda method, url: urllib_exchange(method, url))

    def _open(method: str, url: str):
        return opener(method, url)

    server = Server(
        pin,
        opener=_open,
        spawn=lambda argv, environment, log_path: spawn_ollama(
            argv, environment, log_path, popen=popen or subprocess.Popen
        ),
        stop_tree=lambda pid: stop_process_tree(pid),
        log_path=log_root / f"{run_id}.log",
        bind=assert_port_free if bind is None else bind,
    )
    model = next(arm["model"] for arm in spec["arms"] if arm["id"] == "5090-alone")
    from interlock.adapters.record_fixtures import IDLE_CEILING_MIB, read_card_mib
    from interlock.harness.runner import heartbeat_path

    read_mib = card_mib if card_mib is not None else read_card_mib
    stop_event = threading.Event()
    latest: dict = {}
    latest_lock = threading.Lock()
    companion_thread: dict = {"thread": None}
    sustain_box = {
        "event": threading.Event(),
        "ok": False,
        "reason": None,
        "calls": [],
        "open": None,
        "error": None,
    }
    lane = intel if intel is not None else IntelLane(root, run_id)
    record = SampleRecord()
    gpu_sample = sample if sample is not None else LiveGpuSample(record)
    cpu_sample = cpu if cpu is not None else live_sample_cpu
    burn = CpuBurn()
    every_s = float(spec["measurement"]["steady_sample_ms"]) / 1000.0

    def sample_during(invoke, sample_fn, every):
        if sampler is threaded_sample:
            return threaded_sample(invoke, sample_fn, every, intervals=record.intervals)
        return sampler(invoke, sample_fn, every)

    def call(arm, *, num_predict, measured, kind, prompt, role=None):
        del role
        if arm.get("engine") == "ollama" or arm.get("device") == "5090":

            def invoke():
                payload = owned_generate(post, OWNED, arm, prompt, num_predict)
                normalized = normalize_ollama(payload)
                normalized["measured"] = measured
                normalized["kind"] = kind
                return normalized

            return sample_during(invoke, gpu_sample, every_s)
        return lane.call(arm, num_predict=num_predict, measured=measured, kind=kind, prompt=prompt)

    def _record_latest(payload) -> None:
        if not isinstance(payload, dict):
            return
        with latest_lock:
            latest.clear()
            latest.update(payload)

    def start_companion(arm, *, prompt=""):
        if arm.get("device") == "5090":
            stop_event.clear()
            sustain_box["event"].clear()
            sustain_box["ok"] = False
            sustain_box["reason"] = None
            sustain_box["calls"] = []
            sustain_box["open"] = None
            sustain_box["error"] = None
            with latest_lock:
                latest.clear()

            def run() -> None:
                try:
                    _run_5090_companion(
                        seams,
                        arm,
                        prompt,
                        spec["measurement"],
                        stop_event,
                        sustain_box,
                        _record_latest,
                    )
                except Exception as exc:
                    sustain_box["error"] = exc
                    with latest_lock:
                        latest["error"] = exc
                finally:
                    sustain_box["event"].set()

            thread = threading.Thread(target=run, name="interlock-e4b-companion", daemon=True)
            companion_thread["thread"] = thread
            thread.start()
            return
        lane.start(arm, prompt=prompt)

    def await_sustained_companion(arm):
        del arm
        limit = int(spec["measurement"]["warmup_max_attempts"])
        timeout = (limit + 1) * float(GENERATE_TIMEOUT_S)
        if not sustain_box["event"].wait(timeout):
            stop_event.set()
            raise E4bError("the 5090 companion did not finish its warm-up")
        error = sustain_box.get("error")
        if isinstance(error, BaseException):
            raise error
        return {
            "ok": bool(sustain_box["ok"]),
            "reason": sustain_box.get("reason"),
            "calls": sustain_box.get("calls") or [],
            "open": sustain_box.get("open"),
        }

    def stop_companion(arm, *, kind):
        if kind == "end-generate":
            stop_event.set()
            thread = companion_thread.get("thread")
            if thread is not None:
                thread.join(timeout=120)
                if thread.is_alive():
                    raise E4bError("the 5090 companion generate did not finish")
            return
        if arm.get("engine") == "ollama" or arm.get("device") == "5090":
            unload_owned(post, OWNED, arm["model"])
            return
        lane.stop(arm)
        lane.unload(arm)

    def companion_report(arm) -> dict:
        if arm.get("device") == "5090" or arm.get("engine") == "ollama":
            with latest_lock:
                error = latest.get("error")
                if isinstance(error, BaseException):
                    raise error
                if error:
                    raise E4bError(str(error))
                return dict(latest)
        return lane.report(arm)

    class _Seams:
        pass

    seams = _Seams()
    seams.server = server
    seams.call = call
    seams.start_companion = start_companion
    seams.await_sustained_companion = await_sustained_companion
    seams.stop_companion = stop_companion
    seams.companion_report = companion_report
    seams.sample_cpu = cpu_sample
    seams.sample_gpu = gpu_sample
    seams.sleep = time.sleep
    seams.now = time.perf_counter
    seams.start_threads = burn.start
    seams.stop_threads = burn.stop
    seams.probe = lambda method, path: owned_probe(OWNED, path, _open)

    def load_resident():
        owned_generate(post, OWNED, {"model": model}, "load", 1)

    seams.load_resident = load_resident
    seams.unload_resident = lambda: unload_owned(post, OWNED, model)
    seams.heartbeat = heartbeat_path()
    seams.wall = time.time
    seams.card_models = lambda: card_models_at_ceiling(read_mib, ceiling=IDLE_CEILING_MIB)
    seams.count_tokens = lambda text: count_owned(post, OWNED, model, text)
    seams.git = lambda repo: read_git(repo, git_run or runner)
    seams.vram = read_card_mib if vram is None else vram
    seams.sample_record = record
    seams.probe_timeout_s = PROBE_TIMEOUT_S
    seams.generate_timeout_s = GENERATE_TIMEOUT_S
    seams.transport = post
    seams.trace = []
    seams.resamples = 50000
    return seams
