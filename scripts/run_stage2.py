"""Entry point for Stage 2 distillation (streaming conversion).

Uses the same DistillRecipe but with streaming config enabled.
The config YAML should specify streaming.chunk_sizes, streaming.causal_conv, etc.
"""

import os

# Kill tqdm's TMonitor thread before any other import — it uses anaconda's
# threading.py which causes a GIL crash in multi-process distributed training.
import tqdm.std  # noqa: E402
tqdm.std.TMonitor = type("_NoOpMonitor", (), {
    "__init__": lambda self, *a, **kw: None,
    "exit": lambda self: None,
    "run": lambda self: None,
    "report": lambda self, *a, **kw: None,
})

# Map SLURM env vars to PyTorch distributed before fairseq2 import.
# fairseq2's SlurmHandler sets CUDA_VISIBLE_DEVICES per rank which breaks
# NCCL on clusters that need all GPUs visible. We set the vars manually
# and disable fairseq2's cluster detection via common.cluster=none.
if "SLURM_PROCID" in os.environ:
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("WORLD_SIZE", os.environ["SLURM_NTASKS"])
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ.get("SLURM_NTASKS_PER_NODE", "1"))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

from fairseq2.recipe.cli import train_main

from omni_asr_distil.distill_recipe import DistillRecipe

recipe = DistillRecipe()
train_main(recipe)
