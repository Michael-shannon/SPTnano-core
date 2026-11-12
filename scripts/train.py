"""Train the block3 transformer (contrastive pretraining, then the classifier).

Run in WSL with the GPU (see wsl/train_transformer.sh):
    python scripts/train.py --out_dir /mnt/d/TRANSFORMER_DEVELOPMENT/saved_data/my_new_model
    python scripts/train.py --smoke_test          # 1 epoch on 2,048 windows, to check everything runs

Inputs default to configs/paths.local.yaml (splits_dir, gated_dir). Settings come from configs/pipeline.yaml.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sptnano.config import fix_path, load_paths, load_settings  # noqa: E402
from sptnano.train import train_block3_model  # noqa: E402

if __name__ == "__main__":
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits_dir", default=str(paths["splits_dir"]))
    parser.add_argument("--windowed_parquet", default=str(paths["gated_dir"] / "windowed_df_hierarchical_gates.parquet"))
    parser.add_argument("--out_dir", default=str(paths["training_output_dir"] / "block3_long_large_retrain"))
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    train_block3_model(fix_path(args.splits_dir), fix_path(args.windowed_parquet), fix_path(args.out_dir), load_settings(), smoke_test=args.smoke_test)
