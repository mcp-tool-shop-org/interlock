---
name: read-exit-before-next-gpu-step
description: In a stepped GPU sequence (calibration chain, math ladder), read the finished step's exit code and status before queuing the next step. On 2026-10-09 R&D queued granite while mistral had exited 3 (incomplete) and had to kill it mid-rest.
metadata:
  type: feedback
---

**Read the step's exit code (and "INCOMPLETE") before starting the next model.** A step script that exits 3
means "stop here": resume the same model, or stop and tell the Publisher. Starting the next step is not allowed.

**Why:** 2026-10-09, math ladder. mistral-small:24b exited 3 with one claim stuck. R&D had already launched
granite's step in the same turn as reading the results. It was killed during its rest, before loading, and
TaskStop killed only the wrapper: the `ladder_step.sh` bash and its `sleep 900` survived and had to be killed
by PID. The Publisher's note: "Read the exit code before queuing the next step."

**How to apply:**
- On a step's completion notice, read its `exit` line first, then results, then queue the next step.
- After TaskStop on a background step, check for surviving `ladder_step.sh` / `chain_step.sh` / `sleep`
  processes (PowerShell `Get-CimInstance Win32_Process`) and kill them by PID.
- The grant's stuck-claim rule: one retry, then stop and tell the Publisher.

Related: [[check-ci-after-every-push]], [[thinking-level-is-a-variable-to-weigh]].
