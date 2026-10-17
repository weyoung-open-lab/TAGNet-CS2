"""Holm correction of the paper's significance tests (paper Section IV-K-5, Table XI).

The three test families are corrected separately, as pre-declared in the paper:

* ``context``        - four chi-square tests of contextual variables vs. cluster
                       (``context_trigger_tests.csv``, Table V)
* ``h2_score_state`` - six paired t-tests, leading vs. trailing (``h2_score_state_paired_ttest.csv``, Fig. 8)
* ``h3_opponent``    - per-team chi-square tests of cluster usage vs. opponent (``h3_opponent_chi_square.csv``)

All three input files are written by ``scripts/analyze_results.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "results" / "tagnet_main" / "run_analysis"
FAMILIES = (
    ("context", "context_trigger_tests.csv"),
    ("h2_score_state", "h2_score_state_paired_ttest.csv"),
    ("h3_opponent", "h3_opponent_chi_square.csv"),
)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values, returned in the original order."""
    order = p_values.sort_values(kind="mergesort").index
    m = len(order)
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values.loc[index]))
        running_max = max(running_max, value)
        adjusted.loc[index] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Holm correction within each pre-declared test family.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for family, file_name in FAMILIES:
        frame = pd.read_csv(args.analysis_dir / file_name, encoding="utf-8-sig")
        frame["p_holm"] = holm_adjust(frame["p_value"])
        frame["significant_holm_0_05"] = frame["p_holm"] < args.alpha
        frame["correction_family"] = family
        frame.to_csv(args.output_dir / file_name, index=False, encoding="utf-8-sig")
        summary_rows.append(
            {
                "family": family,
                "file": file_name,
                "test_count": int(len(frame)),
                "significant_raw": int((frame["p_value"] < args.alpha).sum()),
                "significant_holm": int(frame["significant_holm_0_05"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "holm_summary.csv", index=False, encoding="utf-8-sig")
    markdown = (
        "# Holm correction by test family\n\n"
        "Each pre-declared test family was corrected separately.\n\n"
        "```text\n" + summary.to_string(index=False) + "\n```\n"
    )
    (args.output_dir / "holm_summary.md").write_text(markdown, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
