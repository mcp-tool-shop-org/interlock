"""Plan an ``offrig verify calibrate`` invocation.

The command is the one at offrig ``751fb1d`` (``crates/offrig-cli/src/verify_cmd.rs``,
``CalibrateArgs``). ``--project`` is a scratch directory, because opening a project
migrates its store. The URL stays on loopback. ``gpu_cost_hr`` stays 0.

Nothing here launches offrig. A caller that wants a real run passes an executor.
Dry-run and the tests do not.
"""

from __future__ import annotations

from switchyard.adapters.base import AdapterError, StepRequest, StepResult

SCRATCH_ROOT = "E:/AI/switchyard-scratch"
LOOPBACK = frozenset({"http://127.0.0.1:11434", "http://127.0.0.1:11490"})
THINKING = frozenset({"off", "on", "low", "medium", "high"})


def scratch_project(experiment_id: str) -> str:
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise AdapterError("a scratch project needs an experiment id")
    name = experiment_id.strip()
    if any(part in name for part in ("/", "\\", "..")) or name.startswith("."):
        raise AdapterError(f"{experiment_id} is not a single scratch name")
    return f"{SCRATCH_ROOT}/{name}"


def plan_argv(spec: dict) -> list[str]:
    """Argv for ``offrig verify calibrate``. This does not run it."""

    model = spec.get("model")
    if not isinstance(model, str) or not model:
        raise AdapterError("calibrate needs a model")
    lowered = model.lower()
    if ":cloud" in lowered or lowered.endswith("-cloud"):
        raise AdapterError(f"cloud model tags are refused ({model})")
    url = spec.get("url", "http://127.0.0.1:11434")
    if url not in LOOPBACK:
        raise AdapterError("calibrate stays on loopback Ollama")
    if spec.get("gpu_cost_hr", 0) != 0:
        raise AdapterError("gpu_cost_hr must be 0; this adapter does not price a rental")
    gold = spec.get("gold")
    if not isinstance(gold, str) or not gold or ".." in gold.replace("\\", "/").split("/"):
        raise AdapterError("calibrate needs a gold path inside the repo")
    think = spec.get("think", "off")
    if think not in THINKING:
        raise AdapterError(f"think {think!r} is not one offrig accepts")
    num_predict = spec.get("num_predict", 128)
    if not isinstance(num_predict, int) or num_predict < 1:
        raise AdapterError("num_predict must be a positive integer")
    return [
        "offrig",
        "verify",
        "calibrate",
        "--project",
        scratch_project(spec["id"]),
        "--model",
        model,
        "--url",
        url,
        "--think",
        think,
        "--gpu-cost-hr",
        "0",
        "--num-predict",
        str(num_predict),
        gold,
    ]


class OffrigCalibrateAdapter:
    """Builds the argv. Launches only when an executor was passed in."""

    def __init__(self, *, executor=None):
        self.executor = executor

    def run(self, request: StepRequest) -> StepResult:
        spec = dict(request.settings)
        spec.setdefault("id", request.step_id)
        spec.setdefault("model", request.model)
        argv = plan_argv(spec)
        if self.executor is None:
            raise AdapterError("offrig verify calibrate is not launched without an executor")
        code = self.executor(argv)
        if code != 0:
            return StepResult(ok=False, error=f"offrig exited {code}")
        return StepResult(ok=True)

    def unload(self, model: str) -> None:
        return None
