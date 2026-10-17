"""Run every supplementary variant (paper Section IV-K) under five random seeds."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PYTHON = Path(sys.executable)
RUNNER = SCRIPT_DIR / "train_variants.py"
RUN_ROOT = REPO_ROOT / "runs" / "supplementary"
DATA_BASE = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap"
# Prepared data with navmesh geodesic edge distances (columns
# ``distance_nav_geodesic_norm``, ``src_nav_place``, ``dst_nav_place``).
# This dataset is not part of the release; see README, "Known limitations".
DATA_NAV = REPO_ROOT / "data" / "processed" / "dust2_ct_5snap_nav_geodesic"
SEEDS = (42, 137, 271, 503, 911)
VARIANTS = (
    ("full_gat_gru", DATA_BASE, ()),
    ("t20_snapshot", DATA_BASE, ("--single-snapshot", "20")),
    ("mean_pool", DATA_BASE, ("--pool-mode", "mean")),
    ("no_graph_gru", DATA_BASE, ("--no-graph",)),
    ("no_graph_lstm", DATA_BASE, ("--no-graph", "--temporal-cell", "lstm")),
    ("nav_geodesic", DATA_NAV, ("--edge-distance-column", "distance_nav_geodesic_norm")),
    ("nav_geodesic_hinge1", DATA_NAV, ("--edge-distance-column", "distance_nav_geodesic_norm", "--lambda-hinge", "1")),
    ("xy_replace_place", DATA_BASE, ("--node-position-mode", "xy")),
)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log_root = RUN_ROOT / "logs"
    log_root.mkdir(exist_ok=True)
    for name, data_dir, extra_args in VARIANTS:
        if not data_dir.exists():
            print(f"Skipping {name}: prepared data not found at {data_dir}")
            continue
        for seed in SEEDS:
            run_name = f"{name}_seed{seed}"
            run_dir = RUN_ROOT / run_name
            log_path = log_root / f"{run_name}.log"
            if (run_dir / "run_summary.json").exists():
                continue
            command = [
                str(PYTHON), str(RUNNER), "--data-dir", str(data_dir), "--run-root", str(RUN_ROOT),
                "--run-name", run_name, "--seed", str(seed), *extra_args,
            ]
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write("COMMAND: " + " ".join(command) + "\n")
                log_file.flush()
                completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            if completed.returncode:
                raise RuntimeError(f"Training failed: {run_name}; see {log_path}")


if __name__ == "__main__":
    main()
