---
name: git-add-explicit-paths
description: Stage files by exact path, never a parent directory, in any checkout other sessions also write to. On 2026-10-09 R&D's `git add experiments` swept Grok's uncommitted experiments/data-designer/ into a pushed public rnd commit.
metadata:
  type: feedback
---

**Stage by exact file path, never a parent directory, in a checkout another session also writes to.** After
staging, read `git diff --cached --stat` with **no filter** before committing.

**Why:** on 2026-10-09 R&D ran `git add experiments ...` in the rnd checkout. That swept Grok's uncommitted
`experiments/data-designer/` (12 files) into rnd 1.1.4.3.12 (fd2aca3), which was pushed to the public repo.
- The status check before committing used `grep -v data-designer`, so the filter hid exactly what went in.
- The identity scan passed, but it checks identifiers, not credentials. A manual scan found no secrets (a dummy
  env-var key name, loopback endpoints only).
- Still, another session's unreviewed work was published under R&D's commit.

**How to apply:**
- Name each file or the specific new directory you created: `git add experiments/distill-ladder/PLAN.md`, never
  `git add experiments`.
- Never filter the staged list you check before committing. A filter there is exactly where an accident hides.
- If someone else's work does slip into a push, run **both** scans at once before reporting: the identity scan
  (the Director's paths are never posted) and a credentials and endpoints check. Then tell the Director and
  the work's owner. Don't silently rewrite public history. (2026-10-09: the pre-commit identity scan had
  printed CLEAN but wasn't cited; the Publisher's scan of the pushed tarball confirmed 320/320 clean.)

Related: [[gpu-use-through-the-publisher]] (the same multi-session checkout).
