"""Predict behaviour states for a new dataset with the trained model.

    python scripts/predict.py --instant path/to/instant_df_processed.csv --windowed path/to/time_windowed_df_processed.csv

Writes <instant folder>/transformer_predictions/ with:
    window_predictions.parquet         one row per window (probabilities, embedding, behaviour state)
    windowed_with_predictions.parquet  your window table + prediction columns
    instant_with_predictions.parquet   your frame table + prediction columns
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sptnano.config import fix_path, load_paths, load_settings  # noqa: E402
from sptnano.model import load_classifier  # noqa: E402
from sptnano.predict import add_predictions, add_softmax_states, predict_windows, prepare_new_data  # noqa: E402


def read_table(path):
    path = Path(path)
    return pl.read_csv(path, infer_schema_length=10_000) if path.suffix.lower() == ".csv" else pl.read_parquet(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instant", required=True)
    parser.add_argument("--windowed", default=None)
    parser.add_argument("--model_dir", default=str(load_paths()["model_dir"]))
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    instant_path = fix_path(args.instant)
    out_dir = fix_path(args.out_dir) if args.out_dir else instant_path.parent / "transformer_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, class_labels = load_classifier(args.model_dir, device)
    instant = read_table(instant_path).with_columns(pl.col("window_uid").cast(pl.Utf8))
    predictions = predict_windows(prepare_new_data(instant), model, class_labels, device=device)
    behavior = add_softmax_states(predictions, class_labels, **load_settings()["softmax_states"])
    behavior.to_parquet(out_dir / "window_predictions.parquet", index=False)
    add_predictions(instant, behavior).write_parquet(out_dir / "instant_with_predictions.parquet")
    if args.windowed:
        add_predictions(read_table(fix_path(args.windowed)), behavior).write_parquet(out_dir / "windowed_with_predictions.parquet")
    print(behavior["transformer_behavior_state"].value_counts())
    print(f"Saved to {out_dir}")
