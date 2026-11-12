"""Baselines on the transformer's split: random forest on window scalars and a Gaussian HMM.

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --out_dir D:/TRANSFORMER_DEVELOPMENT/saved_data/benchmarks

Writes benchmark_scores.csv (accuracy, balanced accuracy, macro F1 on the held-out test windows).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sptnano.benchmarks import run_benchmarks  # noqa: E402
from sptnano.config import fix_path, load_paths, load_settings  # noqa: E402

if __name__ == "__main__":
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits_dir", default=str(paths["splits_dir"]))
    parser.add_argument("--windowed_parquet", default=str(paths["gated_dir"] / "windowed_df_hierarchical_gates.parquet"))
    parser.add_argument("--out_dir", default=str(paths["evaluation_dir"] / "benchmarks"))
    args = parser.parse_args()

    settings = load_settings()
    run_benchmarks(fix_path(args.splits_dir), fix_path(args.windowed_parquet), fix_path(args.out_dir),
                   label_col=settings["splits"]["label_col"], seed=settings["splits"]["seed"])
