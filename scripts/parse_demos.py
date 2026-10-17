"""Parse CS2 demo files (.dem) into per-match folders read by ``prepare_data.py``.

For every demo, a folder ``<output-root>/<match_id>/`` is written with
``header.json`` and one Parquet file per Awpy table (``rounds``, ``ticks``,
``smokes``, ``infernos``, ...). ``prepare_data.py`` needs ``header.json``,
``rounds``, ``ticks``, ``smokes`` and ``infernos``; the other tables are kept
for further analysis.

The match id is the demo file name without extension, unless a mapping CSV is
given. ``prepare_data.py`` derives the two team names from the folder name, so
folders should follow the pattern ``<team-a>-vs-<team-b>_dust2`` used in
``data/match_list.csv``.

Usage::

    python scripts/parse_demos.py --demo-dir demos/ --output-root data/parsed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "parsed"

# Player properties sampled at every tick. ``prepare_data.py`` uses place, X, Y,
# Z, health, armor, current_equip_value, has_defuser and has_helmet.
PLAYER_PROPS = [
    "X",
    "Y",
    "Z",
    "pitch",
    "yaw",
    "health",
    "armor_value",
    "has_helmet",
    "has_defuser",
    "current_equip_value",
    "active_weapon",
    "inventory",
    "spotted",
    "is_walking",
    "place",
]
TABLES = ("rounds", "ticks", "smokes", "infernos", "kills", "damages", "grenades", "bomb", "shots", "footsteps")


def parse_demo(demo_path: Path, match_dir: Path) -> dict[str, int]:
    from awpy import Demo

    demo = Demo(str(demo_path), verbose=False)
    demo.parse(player_props=PLAYER_PROPS)
    match_dir.mkdir(parents=True, exist_ok=True)
    with (match_dir / "header.json").open("w", encoding="utf-8") as fh:
        json.dump(demo.header, fh, indent=2, default=str)

    row_counts: dict[str, int] = {}
    for name in TABLES:
        table = getattr(demo, name, None)
        if table is None:
            continue
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        frame.to_parquet(match_dir / f"{name}.parquet", index=False)
        row_counts[name] = int(len(frame))
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse CS2 demos into per-match Parquet folders with Awpy.")
    parser.add_argument("--demo-dir", type=Path, required=True, help="Folder searched recursively for *.dem files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--name-map",
        type=Path,
        default=None,
        help="Optional CSV with columns demo_file,match_id to name the output folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    name_map: dict[str, str] = {}
    if args.name_map is not None:
        mapping = pd.read_csv(args.name_map)
        name_map = dict(zip(mapping["demo_file"].astype(str), mapping["match_id"].astype(str)))

    demo_paths = sorted(args.demo_dir.rglob("*.dem"))
    if not demo_paths:
        raise SystemExit(f"No .dem files found under {args.demo_dir}")

    for demo_path in demo_paths:
        match_id = name_map.get(demo_path.name, demo_path.stem)
        match_dir = args.output_root / match_id
        if (match_dir / "ticks.parquet").exists() and not args.overwrite:
            print(f"skip   {match_id} (already parsed)")
            continue
        counts = parse_demo(demo_path, match_dir)
        print(f"parsed {match_id}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
