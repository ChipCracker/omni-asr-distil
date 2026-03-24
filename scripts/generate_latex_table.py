#!/usr/bin/env python
"""Generate LaTeX table for Stage 1 distillation results.

Reads per-sample CSV files from the latest checkpoints and computes
corpus-level WER/CER for the tab:distil_stage1 table in the thesis.

Usage:
    python scripts/generate_latex_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import editdistance

OUTPUT_BASE = Path("/nfs1/scratch/students/witzlch88229/output/distil-stage1")

MODELS = [
    ("S-Large", "s_large_512", r"$\sim$45M"),
    ("S-Medium", "s_medium_384", r"$\sim$28M"),
    ("S-Small", "s_small_256", r"$\sim$15M"),
]

DATASETS = {
    "ort": "rvg1_de",
    "tr2": "rvg1_de_tr2",
}


def find_latest_checkpoint(config_name: str) -> Path | None:
    """Find the highest step_* checkpoint across all ws_* workspaces."""
    output_dir = OUTPUT_BASE / config_name
    if not output_dir.exists():
        return None
    step_dirs = sorted(
        output_dir.glob("ws_*/checkpoints/step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return step_dirs[-1] if step_dirs else None


def compute_corpus_metrics(csv_path: Path) -> tuple[float, float] | None:
    """Compute corpus-level WER and CER from a per-sample CSV."""
    if not csv_path.exists():
        return None

    total_word_err = 0
    total_word_len = 0
    total_char_err = 0
    total_char_len = 0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row["reference"]
            hyp = row["hypothesis"]

            ref_words = ref.split()
            total_word_err += editdistance.eval(hyp.split(), ref_words)
            total_word_len += len(ref_words)
            total_char_err += editdistance.eval(list(hyp), list(ref))
            total_char_len += len(ref)

    if total_word_len == 0:
        return None

    wer = total_word_err / total_word_len * 100
    cer = total_char_err / total_char_len * 100
    return wer, cer


def fmt(val: float | None, is_best: bool = False) -> str:
    if val is None:
        return "--"
    s = f"{val:.2f}"
    return rf"\textbf{{{s}}}" if is_best else s


def main() -> None:
    # Collect results: {config_name: {"ort": (wer, cer), "tr2": (wer, cer)}}
    results: dict[str, dict[str, tuple[float, float] | None]] = {}

    for _, config_name, _ in MODELS:
        checkpoint = find_latest_checkpoint(config_name)
        results[config_name] = {}

        for key, dataset_name in DATASETS.items():
            if checkpoint is None:
                results[config_name][key] = None
                continue

            csv_path = checkpoint / f"eval_{dataset_name}_test.csv"
            results[config_name][key] = compute_corpus_metrics(csv_path)

    # Find best values per column (among students only)
    best = {"ort_wer": None, "ort_cer": None, "tr2_wer": None, "tr2_cer": None}
    for config_name in [m[1] for m in MODELS]:
        for key in DATASETS:
            metrics = results[config_name].get(key)
            if metrics is None:
                continue
            wer, cer = metrics
            bw = f"{key}_wer"
            bc = f"{key}_cer"
            if best[bw] is None or wer < best[bw]:
                best[bw] = wer
            if best[bc] is None or cer < best[bc]:
                best[bc] = cer

    # Generate LaTeX
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(
        r"  \caption{Distillation stage~1: WER and CER (\%) on BAS-RVG1 test sets. "
        r"Teacher: Omnilingual ASR 300M. Best per column in \textbf{bold}.}"
    )
    print(r"  \label{tab:distil_stage1}")
    print(r"  \begin{tabular}{lrcccc}")
    print(r"    \toprule")
    print(
        r"    Student & Params & \multicolumn{2}{c}{ORT} & \multicolumn{2}{c}{TR2} \\"
    )
    print(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    print(r"     & & WER & CER & WER & CER \\")
    print(r"    \midrule")
    print(r"    Teacher (300M) & 300M & -- & -- & -- & -- \\")
    print(r"    \midrule")

    for display_name, config_name, params in MODELS:
        ort = results[config_name].get("ort")
        tr2 = results[config_name].get("tr2")

        ort_wer = ort[0] if ort else None
        ort_cer = ort[1] if ort else None
        tr2_wer = tr2[0] if tr2 else None
        tr2_cer = tr2[1] if tr2 else None

        cols = [
            fmt(ort_wer, ort_wer == best["ort_wer"] if ort_wer is not None else False),
            fmt(ort_cer, ort_cer == best["ort_cer"] if ort_cer is not None else False),
            fmt(tr2_wer, tr2_wer == best["tr2_wer"] if tr2_wer is not None else False),
            fmt(tr2_cer, tr2_cer == best["tr2_cer"] if tr2_cer is not None else False),
        ]

        print(f"    {display_name} & {params} & {' & '.join(cols)} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
