"""Make the train / val / test split from the gated tables (settings: configs/pipeline.yaml -> splits).

    python scripts/make_splits.py --out_dir D:/TRANSFORMER_DEVELOPMENT/data_splits_new
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sptnano.config import fix_path, load_paths, load_settings  # noqa: E402
from sptnano.splits import check_splits, make_splits  # noqa: E402

if __name__ == "__main__":
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gated_dir", default=str(paths["gated_dir"]))
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    gated_dir = fix_path(args.gated_dir)
    make_splits(
        gated_dir / "instant_df_hierarchical_gates.parquet",
        gated_dir / "windowed_df_hierarchical_gates.parquet",
        fix_path(args.out_dir),
        load_settings()["splits"],
    )
    check_splits(fix_path(args.out_dir))
