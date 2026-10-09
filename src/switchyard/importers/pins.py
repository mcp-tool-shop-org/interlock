"""Commit pins for the receipts switchyard imports. Re-check before importing."""

# rnd main, R&D 2026-10-09. Chain and ladder trees match 961f635.
# The two commits after that only add gemma-rerun pre-registration text.
RND_COMMIT = "08ca419f647a3ed69ee645610eaa1c7f35f40f91"

# Kimi's npu-buildout, unmerged. Re-import from the merge commit when the PR merges.
NPU_BUILDOUT_COMMIT = "275e51101f7557bf50898608812cd61d25d5ff17"

# offrig main, the build named in the tie-in. switchyard reads this sha.
# It does not call offrig, and it does not open offrig's database.
OFFRIG_COMMIT = "751fb1d06e600536f91b72e18ce15a8faa4f075e"

# Ladder runs are in RND_COMMIT at math-ladder/results/2026-10-09-ladder/.
# Import those committed copies. Do not read a side directory.

RND_REPO = "E:/AI/Research and Development"
NPU_REPO = "E:/AI/rnd-npu-buildout"

LADDER_RUNS = (
    "ladder-gemma4_31b",
    "ladder-granite4.1_30b",
    "ladder-llama3.1_8b",
    "ladder-mistral-small_24b",
    "ladder-qwen3_14b",
    "ladder-qwen3_8b",
)
