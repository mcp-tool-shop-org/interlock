"""E4b's run: its own Ollama, per-slot warm-up, refusals, and the pilot.

The spec is the contract (experiments/e4b/spec.yaml). This module enforces the
rules the loader used to ignore. Tests pass fakes. Nothing here opens a GPU
or a shared Ollama port unless a caller builds the live seams and runs them.

The 5090 companion is a generate loop. Stopping that loop does not unload
qwen3. The model loads once at block start and unloads once, with keep_alive 0,
at block end. An unload in the middle would reload it and refuse the next slot.

The run-owned server inherits the process environment with every OLLAMA_* and
LLAMA_ARG_* key removed, then the recorded host, models directory and debug
flag. Before it starts, the watchdog heartbeat and an empty card are required.
After the first load, the run refuses unless this start's log names cuda_v13,
size_vram equals size, and the digest matches the spec pin.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from switchyard.guards import require_card_empty, require_grant, require_heartbeat, require_npu_session
from switchyard.harness.e4b_stats import bootstrap_median, decide, effect
from switchyard.harness.e5 import fit_prompt
from switchyard.harness.ollama_serve import (
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
from switchyard.harness.timing import percentile


class E4bError(Exception):
    """E4b cannot run this spec, or a slot broke a rule the receipt must show."""


PILOT_CALLS = 20
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
        except Exception:
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


def slot_refusal(arm: dict, result: dict, *, companion_arm: dict | None, companion: dict | None, tolerance, steady_ok, num_predict) -> str | None:
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


def prepare_prompt(spec: dict, count_tokens) -> str:
    """One prefix for every arm. The counter is the model tokenizer, injected in tests."""

    dataset = spec.get("dataset") or {}
    seed = str(dataset.get("seed") or "")
    target = dataset.get("target_tokens")
    if not seed or isinstance(target, bool) or not isinstance(target, int):
        raise E4bError("E4b fits its prompt from the dataset seed and target")
    if count_tokens is None:
        raise E4bError("E4b fits the prompt with the model tokenizer")
    return fit_prompt(seed, target, count_tokens)


def capture_run(args: list[str]):
    """The live runner: a bare ``subprocess.run`` returns no stdout, so the pin and git reads saw ""."""

    return subprocess.run(args, capture_output=True, text=True, check=False)


def read_git(repo: Path, run) -> dict:
    """HEAD and whether the worktree is clean. The receipt stores the sha, not a path."""

    def _out(args: list[str]) -> str:
        completed = run(args)
        if isinstance(completed, str):
            return completed
        stdout = getattr(completed, "stdout", "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return stdout or ""

    sha = _out(["git", "-C", repo.as_posix(), "rev-parse", "HEAD"]).strip()
    porcelain = _out(["git", "-C", repo.as_posix(), "status", "--porcelain"])
    return {"git_sha": sha, "clean_tree": porcelain.strip() == ""}


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
    if isinstance(size, bool) or isinstance(vram, bool) or not isinstance(size, int) or not isinstance(vram, int) or vram != size:
        raise E4bError("REFUSED: the first load is not fully resident; size_vram must equal size")
    if _digest_hex(entry.get("digest")) != _digest_hex(pin):
        raise E4bError("REFUSED: the model digest does not match the spec pin")


def _require_ready(seams) -> None:
    """Heartbeat, then an empty card. Both run before the server is spawned."""

    path = getattr(seams, "heartbeat", None)
    if path is None:
        from switchyard.harness.runner import heartbeat_path

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


def run_burn(*, target_utility: float, duration_s: float, sample_utility, sample_performance, sleep, start_threads) -> dict:
    """Busy threads matched to a utility target. No GPU work is available here."""

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


def dry_text(spec: dict, blob: str, *, pilot: bool) -> str:
    """What ``switchyard run --dry-run`` prints for E4b. No device is contacted."""

    measurement = spec.get("measurement") or {}
    if pilot:
        lines = [
            "switchyard dry-run: E4b",
            f"spec blob: {blob}",
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
        "switchyard dry-run: E4b",
        f"spec blob: {blob}",
        "own Ollama 127.0.0.1:11441 pinned to the 11434 install",
        "11434 and 11490 not used",
        "probes GET /api/version and GET /api/ps only",
        "session end is keep_alive 0 unload",
        "warm-up: NPU and iGPU one unmeasured call; 5090 768 tokens at least 3 s until steady",
        "until-steady: 10 samples at 200 ms, 2 attempts, else clock_not_steady",
        "companion starts before the warm-up",
        "refusals re-run once in place; two refusals in a pair are inconclusive",
        "Hodges-Lehmann, exact Wilcoxon at least 90%, TOST +/-0.10",
        "extension to 15 when the 10-triplet interval straddles",
        "bootstrap footnote: 50000 resamples, seed 20261012, never decides",
        "burn probe igpu-after-cpu-burn recorded, not a gate",
        tolerance_line,
        band_line,
        "pilot: switchyard run experiments/e4b/spec.yaml --pilot",
        "heartbeat not read; adapters not called",
        "devices not started (--dry-run)",
    ]
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


def _delete_caches(root: Path, relative: list[str]) -> None:
    for rel in relative:
        path = root / rel
        if path.exists():
            shutil.rmtree(path)


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
    """One slot. A paired slot starts its companion before the warm-up."""

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
    for _attempt in range(attempts):
        if companion is not None:
            _trace(seams, ("start_companion", companion["id"]))
            seams.start_companion(companion, prompt=prompt)
        cpu_start = seams.sample_cpu("performance")
        utility = seams.sample_cpu("utility") if arm["device"] == "5090" else None
        steady_ok = None
        if arm["device"] == "5090":
            warm = warm_5090(seams, arm, prompt, measurement, enforce=enforce_steady)
            steady_ok = True if not enforce_steady else warm["ok"]
            if enforce_steady and not warm["ok"]:
                steady_ok = False
        else:
            _trace(seams, ("warm", arm["id"], arm.get("num_predict")))
            seams.call(arm, num_predict=arm.get("num_predict"), measured=False, kind="warm", prompt=prompt)
            warm = {"ok": True, "attempts": []}
        measured = None
        if steady_ok is not False:
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
        cpu_end = seams.sample_cpu("performance")
        report = None
        if companion is not None:
            kind = _stop_kind(companion)
            _trace(seams, ("stop_companion", companion["id"], kind))
            seams.stop_companion(companion, kind=kind)
            report = seams.companion_report(companion)
        if measured is None:
            measured = {"value": None, "clocks_mhz": None, "power_w": None}
        reason = None
        if phase != "cold":
            reason = slot_refusal(
                arm,
                measured,
                companion_arm=companion,
                companion=report if isinstance(report, dict) else None,
                tolerance=tolerance,
                steady_ok=steady_ok if arm["device"] == "5090" and enforce_steady else None,
                num_predict=arm.get("num_predict"),
            )
        if reason:
            events += 1
            reasons.append(reason)
            final = _slot_row(
                arm, role, pair, repeat, measured, reason, events, reasons, cpu_start, cpu_end, utility, root, run_id, warm
            )
            continue
        final = _slot_row(
            arm, role, pair, repeat, measured, None, events, reasons, cpu_start, cpu_end, utility, root, run_id, warm
        )
        break
    if final is None:
        raise E4bError("the slot did not run")
    return final


def _slot_row(arm, role, pair, repeat, measured, reason, events, reasons, cpu_start, cpu_end, utility, root, run_id, warm) -> dict:
    clocks = measured.get("clocks_mhz") if isinstance(measured.get("clocks_mhz"), list) else []
    power = measured.get("power_w") if isinstance(measured.get("power_w"), list) else []
    series_rel = None
    if arm["device"] == "5090" and clocks:
        series = root / "receipts" / f"{run_id}-series" / f"{pair or 'cold'}-{repeat}-{role}-{arm['id']}.jsonl"
        _write_series(series, clocks, power)
        series_rel = series.relative_to(root).as_posix()
    return {
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
        "analysis": role != "cold",
    }


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
            slots.append({"arm": arm_id, "role": role, "pair": item["pair"], "repeat": item["repeat"], "skipped": True, "value": None, "refusal_events": 0})
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


def execute_block(spec: dict, seams, *, root: Path, run_id: str) -> dict:
    """The measured block. Refusals, the estimate, and the burn probe land on the receipt."""

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
            record = _analyze(state["effects"], band=band, resamples=resamples, allow_extension=len(state["effects"]) == 10)
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


def _header(spec, seams, repo, running, caches, prompt, *, aborted: bool) -> dict:
    git = seams.git(repo) if getattr(seams, "git", None) else {"git_sha": None, "clean_tree": None}
    vram = note_vram(getattr(seams, "vram", None))
    header = {
        "kind": "e4b-header",
        "id": spec["id"],
        "git_sha": git.get("git_sha"),
        "clean_tree": git.get("clean_tree"),
        "argv": list(getattr(running, "argv", []) or []),
        "environment": receipt_environment(dict(getattr(running, "environment", {}) or {})),
        "version_start": getattr(running, "version_start", None),
        "version_stop": getattr(running, "version_stop", None),
        "caches": caches,
        "prompt_sha256": _sha256(prompt),
        "aborted": aborted,
    }
    header.update(vram)
    return header


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def run_block(spec: dict, seams, *, root: Path, run_id: str, repo: Path) -> list[dict]:
    """One E4b block. The server's process tree stops on the way out, including on abort."""

    _check_run_id(run_id)
    _require_amended(spec)
    receipt = root / "receipts" / f"{run_id}.jsonl"
    if receipt.exists():
        raise E4bError(f"receipts/{run_id}.jsonl already exists")
    arms = {arm["id"]: arm for arm in spec["arms"]}
    caches = [_make_cache(root, run_id, arm_id) for arm_id in arms]
    rows: list[dict] = []
    started = False
    aborted = False
    running = None
    prompt = ""
    _require_ready(seams)
    try:
        with seams.server as running:
            started = True
            _probe(seams, "GET", "/api/version")
            _probe(seams, "GET", "/api/ps")
            try:
                seams.load_resident()
                _require_owned_server(spec, seams, running)
                body = execute_block(spec, seams, root=root, run_id=run_id)
                rows = _rows_of(body)
                prompt = body["prompt"]
            finally:
                try:
                    seams.unload_resident()
                finally:
                    _probe(seams, "GET", "/api/version")
    except Exception:
        aborted = True
        raise
    finally:
        _delete_caches(root, caches)
        if started and running is not None:
            header = _header(spec, seams, repo, running, caches, prompt, aborted=aborted)
            _write_jsonl(receipt, [header, *rows])
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


def run_pilot(spec: dict, seams, *, root: Path, run_id: str, repo: Path, calls: int = PILOT_CALLS) -> dict:
    """5090-alone only. Rows go to pilot-e4b/ and never into receipts/ or the table."""

    _check_run_id(run_id)
    destination = pilot_destination(root, run_id)
    if destination.exists():
        raise E4bError(f"pilot file {destination.name} already exists")
    arm = next((item for item in spec["arms"] if item["id"] == "5090-alone"), None)
    if arm is None:
        raise E4bError("the pilot measures 5090-alone")
    measurement = _measurement(spec)
    prompt = prepare_prompt(spec, seams.count_tokens)
    ranges = []
    full = []
    warm_short = 0
    _require_ready(seams)
    with seams.server as running:
        _probe(seams, "GET", "/api/version")
        _probe(seams, "GET", "/api/ps")
        try:
            seams.load_resident()
            _require_owned_server(spec, seams, running)
            for _index in range(calls):
                warm = warm_5090(seams, arm, prompt, measurement, enforce=False)
                if not warm["attempts"] or float(warm["attempts"][-1].get("elapsed_s") or 0) + 1e-9 < float(measurement["warmup_min_s"]):
                    warm_short += 1
                measured = dict(
                    seams.call(arm, num_predict=arm.get("num_predict"), measured=True, kind="measure", prompt=prompt)
                )
                clocks = measured.get("clocks_mhz")
                if not isinstance(clocks, list) or not clocks or any(item is None for item in clocks):
                    raise E4bError("the pilot is missing a within-call clock range")
                ranges.append(float(max(clocks) - min(clocks)))
                count = measured.get("eval_count")
                predict = arm.get("num_predict")
                full.append(isinstance(count, int) and not isinstance(count, bool) and count >= predict)
        finally:
            seams.unload_resident()
    summary = pilot_summary(ranges, full, step=float(measurement.get("clock_step_mhz") or CLOCK_STEP_MHZ))
    git = seams.git(repo) if getattr(seams, "git", None) else {"git_sha": None, "clean_tree": None}
    document = {
        "kind": "e4b-pilot",
        "calls": calls,
        "num_predict": arm.get("num_predict"),
        "warmup_num_predict": measurement.get("warmup_num_predict"),
        "warmups_under_min_s": warm_short,
        "ranges_mhz": ranges,
        "git_sha": git.get("git_sha"),
        "clean_tree": git.get("clean_tree"),
        "argv": list(getattr(running, "argv", []) or []),
        "environment": receipt_environment(dict(getattr(running, "environment", {}) or {})),
        "version_start": getattr(running, "version_start", None),
        "version_stop": getattr(running, "version_stop", None),
    }
    document.update(summary)
    document.update(note_vram(getattr(seams, "vram", None)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
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
) -> list | dict:
    """Flag checks, then the pilot or the block. Null tolerances refuse before any device."""

    from switchyard.harness.lock import require_unmodified

    require_unmodified(repo, spec_path)
    require_grant(grant, uses_5090=True)
    if not pilot:
        require_npu_session(npu_session, device="NPU")
        require_npu_session(npu_session, device="iGPU")
        _require_amended(spec)
    name = run_id or ("e4b-pilot" if pilot else "e4b-block")
    if seams is None:
        seams = build_live_seams(root, name, spec, pilot=pilot)
    if pilot:
        return run_pilot(spec, seams, root=root, run_id=name, repo=repo)
    rows = run_block(spec, seams, root=root, run_id=name, repo=repo)
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
    return {
        "value": value,
        "load_duration_s": load_s,
        "eval_count": eval_count if isinstance(eval_count, int) and not isinstance(eval_count, bool) else None,
        "elapsed_s": elapsed,
        "truncated_tokens": None,
    }


def threaded_sample(invoke, sample, every_s: float) -> dict:
    """Sample while ``invoke`` runs. Tests that need a deterministic series skip this."""

    series: list[tuple] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            series.append(tuple(sample()))
            if stop.wait(every_s):
                break

    thread = threading.Thread(target=loop, name="switchyard-e4b-sample", daemon=True)
    thread.start()
    try:
        result = dict(invoke())
    finally:
        stop.set()
        thread.join(timeout=every_s + 1)
    if "clocks_mhz" not in result:
        result["clocks_mhz"] = [item[0] for item in series]
        result["power_w"] = [item[1] for item in series]
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


def spawn_ollama(argv: list[str], environment: dict, log_path: Path, *, popen=subprocess.Popen, system: str | None = None):
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


def urllib_exchange(method: str, url: str, body: dict | None = None) -> dict:
    """The live HTTP call. Tests pass their own transport and never reach this."""

    import urllib.request

    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise E4bError("Ollama returned a non-object")
    return payload


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
            from switchyard.adapters.openvino import OpenvinoAdapter

            self._adapter = OpenvinoAdapter()
        return self._adapter

    def request(self, arm: dict, prompt: str):
        from switchyard.adapters.base import StepRequest

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


def build_live_seams(root: Path, run_id: str, spec: dict, *, pilot: bool, run=None, popen=None, transport=None, opener=None, sample=None, cpu=None, intel=None, sampler=threaded_sample, vram=None, git_run=None, bind=None, card_mib=None):
    """Wire the run-owned server. Every external call is injectable; the defaults are the live ones."""

    from switchyard.harness.ollama_serve import Server

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
    from switchyard.adapters.record_fixtures import IDLE_CEILING_MIB, read_card_mib
    from switchyard.harness.runner import heartbeat_path

    read_mib = card_mib if card_mib is not None else read_card_mib
    stop_event = threading.Event()
    latest: dict = {}
    latest_lock = threading.Lock()
    companion_thread: dict = {"thread": None}
    lane = intel if intel is not None else IntelLane(root, run_id)

    def call(arm, *, num_predict, measured, kind, prompt, role=None):
        del role
        if arm.get("engine") == "ollama" or arm.get("device") == "5090":

            def invoke():
                payload = owned_generate(post, OWNED, arm, prompt, num_predict)
                normalized = normalize_ollama(payload)
                normalized["measured"] = measured
                normalized["kind"] = kind
                return normalized

            return sampler(
                invoke,
                sample or (lambda: (None, None)),
                float(spec["measurement"]["steady_sample_ms"]) / 1000.0,
            )
        return lane.call(arm, num_predict=num_predict, measured=measured, kind=kind, prompt=prompt)

    def start_companion(arm, *, prompt=""):
        if arm.get("device") == "5090":
            stop_event.clear()
            with latest_lock:
                latest.clear()

            def post_arm(item):
                payload = call(
                    item,
                    num_predict=item.get("num_predict"),
                    measured=False,
                    kind="companion",
                    prompt=prompt,
                )
                if isinstance(payload, dict):
                    with latest_lock:
                        latest.clear()
                        latest.update(payload)
                return payload

            def run() -> None:
                try:
                    generate_loop(arm, post_arm, stop_event)
                except Exception as exc:
                    with latest_lock:
                        latest["error"] = str(exc)

            thread = threading.Thread(target=run, name="switchyard-e4b-companion", daemon=True)
            companion_thread["thread"] = thread
            thread.start()
            return
        lane.start(arm, prompt=prompt)

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
                if latest.get("error"):
                    raise E4bError(str(latest["error"]))
                return dict(latest)
        return lane.report(arm)

    class _Seams:
        pass

    seams = _Seams()
    seams.server = server
    seams.call = call
    seams.start_companion = start_companion
    seams.stop_companion = stop_companion
    seams.companion_report = companion_report
    seams.sample_cpu = cpu or (lambda name: None)
    seams.sample_gpu = sample or (lambda: (None, None))
    seams.sleep = time.sleep
    seams.now = time.perf_counter
    seams.start_threads = lambda count: None
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
    seams.vram = vram
    seams.trace = []
    seams.resamples = 50000
    return seams
