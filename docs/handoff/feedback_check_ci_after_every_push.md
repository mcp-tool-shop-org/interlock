---
name: check-ci-after-every-push
description: After every push to rnd (or any repo), wait for that commit's CI result before calling the work done. Twice on 2026-10-08/09 rnd CI sat red for several commits (Atlas map drift) unnoticed.
metadata:
  type: feedback
---

**After every push, wait for that commit's CI to finish and read the result.** "Pushed" is not "done".

**Why:**
- On 2026-10-08/09, rnd's CI went red twice without R&D noticing: 8 commits from 00f57b4, then 3 from fd2aca3.
  Both times the cause was Atlas structure drift, from new Python files or new tests→experiments edges that the
  committed map didn't have.
- The rule `atlas-map.md` says: regenerate and commit `atlas/` with the change. Each fix took two minutes once
  seen. The cost was the commits that went out red in between.

**How to apply:**
- After `git push`, poll `gh run list --workflow CI --limit 1` until that SHA completes. On failure, read
  `gh run view <id> --log-failed`.
- Expect Atlas drift whenever a change adds Python modules, tests that import from new places, or new
  directories. Regenerate before pushing:
  - `git worktree add --detach <tmp> HEAD`;
  - `npx --yes @dogfood-lab/atlas@1.24.0 map` there (a clean tree, so other sessions' untracked files don't
    leak in);
  - copy `atlas/` back.

Related: [[git-add-explicit-paths]].
