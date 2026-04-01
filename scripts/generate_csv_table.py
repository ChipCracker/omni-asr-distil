#!/usr/bin/env python
"""Generate CSV summary tables for Stage 1 and Stage 2 distillation results.

Usage:
    python scripts/generate_csv_table.py          # both tables
    python scripts/generate_csv_table.py --stage1  # stage 1 only
    python scripts/generate_csv_table.py --stage2  # stage 2 only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import editdistance

STAGE1_OUTPUT = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage1")
STAGE2_OUTPUT = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage2")

STAGE1_MODELS = [
    ("S-Large", "s_large_512", "~45M"),
    ("S-Medium", "s_medium_384", "~28M"),
    ("S-Small", "s_small_256", "~15M"),
]

STAGE2_CONFIGS = [
    ("S-Large", [
        ("Non-streaming (stage 1)", None, "s_large_512"),
        ("DCT", "stream_dct_large", None),
    ]),
    ("S-Medium", [
        ("Non-streaming (stage 1)", None, "s_medium_384"),
        ("DCT", "stream_dct", None),
    ]),
]

DATASETS = {
    "ort": "rvg1_de",
    "tr2": "rvg1_de_tr2",
}


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    step_dirs = sorted(
        output_dir.glob("ws_*/checkpoints/step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return step_dirs[-1] if step_dirs else None


def compute_corpus_metrics(csv_path: Path) -> tuple[float, float] | None:
    if not csv_path.exists():
        return None
    total_word_err = total_word_len = total_char_err = total_char_len = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ref, hyp = row["reference"], row["hypothesis"]
            ref_words = ref.split()
            total_word_err += editdistance.eval(hyp.split(), ref_words)
            total_word_len += len(ref_words)
            total_char_err += editdistance.eval(list(hyp), list(ref))
            total_char_len += len(ref)
    if total_word_len == 0:
        return None
    return total_word_err / total_word_len * 100, total_char_err / total_char_len * 100


def get_metrics(checkpoint: Path | None, dataset_name: str) -> tuple[float, float] | None:
    if checkpoint is None:
        return None
    return compute_corpus_metrics(checkpoint / f"eval_{dataset_name}_test.csv")


def fmt(val: float | None) -> str:
    return f"{val:.1f}" if val is not None else ""


def generate_stage1():
    writer = csv.writer(sys.stdout)
    writer.writerow(["Student", "Params", "ORT WER", "ORT CER", "TR2 WER", "TR2 CER"])

    for display_name, config_name, params in STAGE1_MODELS:
        checkpoint = find_latest_checkpoint(STAGE1_OUTPUT / config_name)
        ort = get_metrics(checkpoint, DATASETS["ort"])
        tr2 = get_metrics(checkpoint, DATASETS["tr2"])
        writer.writerow([
            display_name, params,
            fmt(ort[0] if ort else None), fmt(ort[1] if ort else None),
            fmt(tr2[0] if tr2 else None), fmt(tr2[1] if tr2 else None),
        ])


def generate_stage2():
    writer = csv.writer(sys.stdout)
    writer.writerow(["Student", "Streaming Config", "ORT WER", "ORT CER"])

    for student_name, configs in STAGE2_CONFIGS:
        for stream_name, stage2_config, stage1_config in configs:
            if stage1_config is not None:
                checkpoint = find_latest_checkpoint(STAGE1_OUTPUT / stage1_config)
            else:
                checkpoint = find_latest_checkpoint(STAGE2_OUTPUT / stage2_config)
            metrics = get_metrics(checkpoint, "rvg1_de")
            writer.writerow([
                student_name, stream_name,
                fmt(metrics[0] if metrics else None),
                fmt(metrics[1] if metrics else None),
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    args = parser.parse_args()

    both = not args.stage1 and not args.stage2

    if both or args.stage1:
        if both:
            print("# Stage 1")
        generate_stage1()
    if both:
        print()
    if both or args.stage2:
        if both:
            print("# Stage 2")
        generate_stage2()


if __name__ == "__main__":
    main()
