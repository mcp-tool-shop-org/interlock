---
name: gpu-use-through-the-publisher
description: The local GPU (RTX 5090) may be used whenever it is free and watched, but tell the Publisher session first, so one session always knows the GPU's status (Director, 2026-10-07).
metadata:
  type: feedback
---

**Use the 5090 when it is free and you watch it, and tell the Publisher first.** Given by the Director on 2026-10-07, while the planted-defects program was queuing GPU work (Step B detector heads, a Kev fine-tune) behind aspire-si.

**Why:** several sessions share one card (aspire-si, R&D, ai-jam-sessions, translations). One session has to know at any moment who holds the GPU, or work collides and the VRAM watchdog kills a process. The Publisher is that session.

**How to apply:**
- Before any GPU job: check it is free (`nvidia-smi`; the VRAM watchdog's state, see [[vram-watchdog]]). Then message the Publisher with what will run, roughly how long, and the expected VRAM. Start only after that.
- While it runs, watch it: memory caps on (e.g. a PyTorch memory fraction of 0.82 on the 32 GB card), and the watchdog alive.
- When it ends, or is stopped, tell the Publisher the card is free again.
- No separate go from the Director is needed per job once this holds. Paid pods still go through offrig's budget, which only he sets.
- Pass this on to any peer session that queues GPU work (R&D, aspire-si).

**Tightened 2026-10-08 (Mike, directly to the Publisher, relayed to every GPU-using session):** "taking it now" is no longer enough. Ask the Publisher with the job, VRAM and duration, **wait for its explicit "granted"**, then run, then send "card free". One holder at a time, to avoid races. Still no per-job go from Mike.

**Rest rule, 2026-10-08 (Mike, directly to the Publisher):** the GPU rests **15 minutes with nothing loaded
between runs.**
- In a multi-model chain: unload (keep_alive 0, confirm `/api/ps` is empty), wait 15 minutes, then load the
  next. Put the rests in any overnight sizing.
- The Publisher's grants follow it too: the next slot starts no sooner than 15 minutes after a "card free".
- The Publisher's reading: a short burst of 1–3 minute loads (a ~20-minute smoke) counts as one run, with one
  rest before it and one after. Every sustained run gets its own rest.
- A health monitor logs the card every 30 s overnight and may pause a run, so chains must stop on an
  incomplete run and resume cleanly. rnd `experiments/verifier-gold/calibration/chain_step.sh` does this.
