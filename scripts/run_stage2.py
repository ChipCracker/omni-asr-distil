"""Entry point for Stage 2 distillation (streaming conversion).

Uses the same DistillRecipe but with streaming config enabled.
The config YAML should specify streaming.chunk_sizes, streaming.causal_conv, etc.
"""

# Disable tqdm monitor thread before any imports (prevents GIL crash with torchrun)
from tqdm import tqdm
tqdm.monitor_interval = 0

from fairseq2.recipe.cli import train_main

from omni_asr_distil.distill_recipe import DistillRecipe

recipe = DistillRecipe()
train_main(recipe)
