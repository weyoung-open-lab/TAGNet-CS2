"""Build the Dust2 base map used under the heatmaps of Fig. 3.

The base map is the walkable area of de_dust2 estimated from player positions:
the (X, Y) positions of all players at every tick of the given ``ticks.parquet``
files (written by ``parse_demos.py``) are binned on a 16-unit grid, cells with
at least two samples are marked walkable, and small gaps are closed. No game
asset is used, so the result can be redistributed with the repository.

The released mask ``data/map/dust2_walkable_mask.npz`` was built from three
parsed professional Dust2 matches (about 5 million player-tick positions).

Usage::

    python scripts/build_map_mask.py data/parsed/*/ticks.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "map" / "dust2_walkable_mask.npz"
CELL = 16
X_RANGE = (-2600, 2000)
Y_RANGE = (-1300, 3500)
MIN_SAMPLES = 2
MIN_GAP_CELLS = 60


def build_mask(positions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_edges = np.arange(X_RANGE[0], X_RANGE[1] + CELL, CELL)
    y_edges = np.arange(Y_RANGE[0], Y_RANGE[1] + CELL, CELL)
    counts, _, _ = np.histogram2d(positions["X"], positions["Y"], bins=[x_edges, y_edges])
    mask = counts >= MIN_SAMPLES
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = ndimage.binary_opening(mask, iterations=1)
    gaps, n_gaps = ndimage.label(~mask)
    gap_sizes = ndimage.sum(~mask, gaps, range(1, n_gaps + 1))
    for gap_id, size in enumerate(gap_sizes, start=1):
        if size < MIN_GAP_CELLS:
            mask[gaps == gap_id] = True
    return mask, x_edges, y_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the Dust2 walkable area from parsed tick positions.")
    parser.add_argument("ticks", nargs="+", type=Path, help="ticks.parquet files written by parse_demos.py")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    positions = pd.concat(
        [pd.read_parquet(path, columns=["X", "Y"]) for path in args.ticks], ignore_index=True
    ).dropna()
    mask, x_edges, y_edges = build_mask(positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, mask=mask, x_edges=x_edges, y_edges=y_edges)
    print(f"{len(positions):,} positions, {mask.sum():,} walkable cells -> {args.output}")


if __name__ == "__main__":
    main()
