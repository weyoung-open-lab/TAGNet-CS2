#!/usr/bin/env bash
# Rebuild the main tables and figures of the paper from the released checkpoint.
# Outputs go to runs/reproduce/; results/ is left untouched.
set -euo pipefail
cd "$(dirname "$0")"

OUT=runs/reproduce
mkdir -p "$OUT"
rm -rf "$OUT/run" "$OUT/run_analysis"
cp -R results/tagnet_main/run "$OUT/run"

python scripts/analyze_results.py --run-dir "$OUT/run"
python scripts/visualize_results.py --analysis-dir "$OUT/run_analysis"
python scripts/extract_attention.py --run-dir "$OUT/run" --analysis-dir "$OUT/run_analysis"
python scripts/team_fingerprints_by_name.py --analysis-dir "$OUT/run_analysis"
python scripts/ablation_table.py --output "$OUT/ablation_table.csv"
python supplementary/rule_baseline.py --data-dir data/processed/dust2_ct_5snap \
    --tagnet-run-dir "$OUT/run" --output-dir "$OUT/rule_baseline"
python supplementary/holm_correction.py --analysis-dir "$OUT/run_analysis" --output-dir "$OUT/holm"
python supplementary/summarize_supplement.py --run-root results/supplementary/runs \
    --output-dir "$OUT/supplementary_summary"

python - "$OUT" <<'EOF'
import sys
import pandas as pd

out = sys.argv[1]
k = pd.read_csv(f"{out}/run_analysis/k_selection_metrics.csv", encoding="utf-8-sig")
best = k.sort_values(["avg_rank", "rank_silhouette", "k"]).iloc[0]
chi = pd.read_csv(f"{out}/run_analysis/context_trigger_tests.csv", encoding="utf-8-sig").set_index("variable")
sizes = pd.read_parquet(f"{out}/run_analysis/analysis_cluster_assignments_bestk.parquet")["cluster_id"].value_counts().sort_index()
print(f"Selected K = {int(best.k)} (mean rank {best.avg_rank:.2f}); paper: K = 6 (4.33)")
print(f"Cluster sizes C0-C5 = {sizes.tolist()}; paper: [124, 109, 115, 151, 136, 62]")
print(f"Chi-square, CT economy type = {chi.loc['ct_econ_type', 'chi2']:.2f}; paper: 96.25")
EOF
