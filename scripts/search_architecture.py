from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TRAIN_SCRIPT = SCRIPT_DIR / "train_tagnet.py"
ANALYZE_SCRIPT = SCRIPT_DIR / "analyze_results.py"
VIS_SCRIPT = SCRIPT_DIR / "visualize_results.py"
ATTN_SCRIPT = SCRIPT_DIR / "extract_attention.py"
RUN_ROOT = PROJECT_ROOT / "runs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dust2_ct_5snap"


SEARCH_CONFIGS = [
    {"gat_layers": 1, "gru_layers": 1, "hidden_dim": 64, "embedding_dim": 64},
    {"gat_layers": 2, "gru_layers": 1, "hidden_dim": 64, "embedding_dim": 64},
    {"gat_layers": 3, "gru_layers": 1, "hidden_dim": 64, "embedding_dim": 64},
    {"gat_layers": 2, "gru_layers": 1, "hidden_dim": 32, "embedding_dim": 32},
    {"gat_layers": 2, "gru_layers": 1, "hidden_dim": 64, "embedding_dim": 32},
    {"gat_layers": 2, "gru_layers": 1, "hidden_dim": 64, "embedding_dim": 128},
    {"gat_layers": 2, "gru_layers": 1, "hidden_dim": 128, "embedding_dim": 64},
    {"gat_layers": 2, "gru_layers": 2, "hidden_dim": 64, "embedding_dim": 64},
]


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def config_name(cfg: dict[str, int]) -> str:
    return f"g{cfg['gat_layers']}_gru{cfg['gru_layers']}_h{cfg['hidden_dim']}_e{cfg['embedding_dim']}"


def train_and_analyze(
    python_exe: str,
    data_dir: Path,
    run_name: str,
    cfg: dict[str, int],
    epochs: int,
    patience: int,
    batch_size: int,
    reuse_existing: bool,
) -> tuple[Path, Path]:
    run_dir = RUN_ROOT / run_name
    analysis_dir = Path(f"{run_dir}_analysis")
    if not (reuse_existing and (run_dir / "run_summary.json").exists()):
        cmd = [
            python_exe,
            str(TRAIN_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--run-name",
            run_name,
            "--epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--batch-size",
            str(batch_size),
            "--k-list",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "--hidden-dim",
            str(cfg["hidden_dim"]),
            "--embedding-dim",
            str(cfg["embedding_dim"]),
            "--gat-layers",
            str(cfg["gat_layers"]),
            "--gru-layers",
            str(cfg["gru_layers"]),
        ]
        run_cmd(cmd, PROJECT_ROOT)
    if not (reuse_existing and (analysis_dir / "analysis_summary.json").exists()):
        cmd = [
            python_exe,
            str(ANALYZE_SCRIPT),
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--min-k",
            "2",
            "--max-k",
            "15",
        ]
        run_cmd(cmd, PROJECT_ROOT)
    return run_dir, analysis_dir


def collect_run_metrics(cfg: dict[str, int], run_dir: Path, analysis_dir: Path) -> dict[str, object]:
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    analysis_summary = json.loads((analysis_dir / "analysis_summary.json").read_text(encoding="utf-8"))
    k_df = pd.read_csv(analysis_dir / "k_selection_metrics.csv")
    chosen_k = int(analysis_summary["recommended_k_from_multi_metric_analysis"])
    chosen_row = k_df[k_df["k"] == chosen_k].iloc[0]
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "gat_layers": cfg["gat_layers"],
        "gru_layers": cfg["gru_layers"],
        "hidden_dim": cfg["hidden_dim"],
        "embedding_dim": cfg["embedding_dim"],
        "train_best_k": int(run_summary["best_k_by_silhouette"]),
        "analysis_best_k": chosen_k,
        "train_loss": float(run_summary["train_eval"]["loss"]),
        "val_loss": float(run_summary["val_eval"]["loss"]),
        "test_loss": float(run_summary["test_eval"]["loss"]),
        "silhouette": float(chosen_row["silhouette"]),
        "davies_bouldin": float(chosen_row["davies_bouldin"]),
        "calinski_harabasz": float(chosen_row["calinski_harabasz"]),
        "weak_label_purity_at_15s": float(chosen_row["weak_label_purity_at_15s"]) if pd.notna(chosen_row["weak_label_purity_at_15s"]) else None,
        "min_cluster_size": int(chosen_row["min_cluster_size"]),
        "max_cluster_size": int(chosen_row["max_cluster_size"]),
        "cluster_sizes": str(chosen_row["cluster_sizes"]),
    }


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_silhouette"] = df["silhouette"].rank(ascending=False, method="dense")
    df["rank_davies_bouldin"] = df["davies_bouldin"].rank(ascending=True, method="dense")
    df["rank_calinski_harabasz"] = df["calinski_harabasz"].rank(ascending=False, method="dense")
    df["rank_val_loss"] = df["val_loss"].rank(ascending=True, method="dense")
    df["rank_test_loss"] = df["test_loss"].rank(ascending=True, method="dense")
    df["avg_rank"] = df[
        [
            "rank_silhouette",
            "rank_davies_bouldin",
            "rank_calinski_harabasz",
            "rank_val_loss",
            "rank_test_loss",
        ]
    ].mean(axis=1)
    return df.sort_values(["avg_rank", "rank_silhouette", "rank_test_loss", "gat_layers", "hidden_dim"])


def run_final_bundle(python_exe: str, data_dir: Path, cfg: dict[str, int], final_run_name: str, final_epochs: int, final_patience: int, batch_size: int) -> tuple[Path, Path]:
    run_dir = RUN_ROOT / final_run_name
    analysis_dir = Path(f"{run_dir}_analysis")

    train_cmd = [
        python_exe,
        str(TRAIN_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--run-name",
        final_run_name,
        "--epochs",
        str(final_epochs),
        "--patience",
        str(final_patience),
        "--batch-size",
        str(batch_size),
        "--k-list",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--embedding-dim",
        str(cfg["embedding_dim"]),
        "--gat-layers",
        str(cfg["gat_layers"]),
        "--gru-layers",
        str(cfg["gru_layers"]),
    ]
    run_cmd(train_cmd, PROJECT_ROOT)
    run_cmd([python_exe, str(ANALYZE_SCRIPT), "--data-dir", str(data_dir), "--run-dir", str(run_dir), "--min-k", "2", "--max-k", "15"], PROJECT_ROOT)
    run_cmd([python_exe, str(VIS_SCRIPT), "--data-dir", str(data_dir), "--analysis-dir", str(analysis_dir)], PROJECT_ROOT)
    run_cmd([python_exe, str(ATTN_SCRIPT), "--data-dir", str(data_dir), "--run-dir", str(run_dir), "--analysis-dir", str(analysis_dir)], PROJECT_ROOT)
    return run_dir, analysis_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Search TAGNet architecture and retrain the best configuration.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-prefix", default="tagnet")
    parser.add_argument("--search-epochs", type=int, default=25)
    parser.add_argument("--search-patience", type=int, default=6)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--final-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    timestamp = datetime.now().strftime("%Y%m%d")
    report_dir = RUN_ROOT / f"{args.run_prefix}_architecture_search_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cfg in SEARCH_CONFIGS:
        name = f"{args.run_prefix}_search_{config_name(cfg)}"
        run_dir, analysis_dir = train_and_analyze(
            python_exe=args.python_exe,
            data_dir=data_dir,
            run_name=name,
            cfg=cfg,
            epochs=args.search_epochs,
            patience=args.search_patience,
            batch_size=args.batch_size,
            reuse_existing=args.reuse_existing,
        )
        rows.append(collect_run_metrics(cfg, run_dir, analysis_dir))

    results_df = add_ranks(pd.DataFrame(rows))
    results_df.to_csv(report_dir / "architecture_search_results.csv", index=False, encoding="utf-8-sig")

    best_row = results_df.iloc[0]
    best_cfg = {
        "gat_layers": int(best_row["gat_layers"]),
        "gru_layers": int(best_row["gru_layers"]),
        "hidden_dim": int(best_row["hidden_dim"]),
        "embedding_dim": int(best_row["embedding_dim"]),
    }
    final_run_name = f"{args.run_prefix}_best_{config_name(best_cfg)}_{timestamp}"
    final_run_dir, final_analysis_dir = run_final_bundle(
        python_exe=args.python_exe,
        data_dir=data_dir,
        cfg=best_cfg,
        final_run_name=final_run_name,
        final_epochs=args.final_epochs,
        final_patience=args.final_patience,
        batch_size=args.batch_size,
    )

    summary = {
        "search_configs": SEARCH_CONFIGS,
        "best_config": best_cfg,
        "best_search_run": str(best_row["run_dir"]),
        "final_run_dir": str(final_run_dir),
        "final_analysis_dir": str(final_analysis_dir),
    }
    (report_dir / "architecture_search_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown = [
        "# TAGNet Architecture Search",
        "",
        f"- Best config: `gat_layers={best_cfg['gat_layers']}`, `gru_layers={best_cfg['gru_layers']}`, `hidden_dim={best_cfg['hidden_dim']}`, `embedding_dim={best_cfg['embedding_dim']}`",
        f"- Final run: `{final_run_dir}`",
        "",
        "## Ranked Results",
        "",
        "```text",
        results_df.to_string(index=False),
        "```",
        "",
    ]
    (report_dir / "architecture_search_summary.md").write_text("\n".join(markdown), encoding="utf-8")

    print(results_df.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
