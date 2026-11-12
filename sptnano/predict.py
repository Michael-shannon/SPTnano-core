"""Step 6: run the trained model on windows and turn the softmax into behaviour states.

predict_windows      one row per window: probabilities (prob_<state>), predicted label, 16-D embedding (emb_00..15)
add_softmax_states   top1/top2, margin, entropy and the final transformer_behavior_state:
                       top1 >= 0.70                                   -> that state (e.g. "bound")
                       top1 <= 0.70, top2 >= 0.20, top1-top2 <= 0.40  -> the pair (e.g. "bound_confined")
                       otherwise                                      -> "unresolved_<top1>"
add_predictions      left-join the new columns onto any table with window_uid (window or frame level)
"""

from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .tokens import WindowDataset

WINDOW_METADATA_COLUMNS = [
    "window_uid", "unique_id", "filename", "condition", "condition_comp", "replicate", "molecule", "mol", "cell",
    "type", "location", "geno", "group", "cas", "dox", "nocodazole_concentration",
    "final_population", "final_population_certain",
]
PREDICTION_DEFAULTS = {"filename": "new_data", "condition": "new_data", "replicate": "new_data", "molecule": "new_data"}


def heldout_test_frames(splits_dir, instant_parquet):
    """Frame rows of every held-out test window (test cells + the forced replicate), labelled or not."""
    with open(f"{splits_dir}/data_splits.pkl", "rb") as f:
        split_data = pickle.load(f)
    test_cells = list(split_data["split_info"]["test_cells"])
    test_replicate = split_data["split_config"].get("force_test_replicate_value", "R3")
    schema = pl.scan_parquet(instant_parquet).collect_schema().names()
    columns = [c for c in ["window_uid", "frame", "x_um", "y_um", "direction_rad"] + WINDOW_METADATA_COLUMNS if c in schema]
    columns = list(dict.fromkeys(columns))
    return (
        pl.scan_parquet(instant_parquet)
        .filter(pl.col("filename").is_in(test_cells) | (pl.col("replicate") == test_replicate))
        .select(columns)
        .collect()
    )


def prepare_new_data(instant_df):
    """Fill the metadata columns the prediction output expects, for data that is not a master table."""
    out = instant_df.with_columns(pl.col("window_uid").cast(pl.Utf8))
    if "molecule" not in out.columns and "mol" in out.columns:
        out = out.with_columns(pl.col("mol").cast(pl.Utf8).alias("molecule"))
    if "unique_id" not in out.columns:
        out = out.with_columns(pl.col("window_uid").alias("unique_id"))
    for col, default in PREDICTION_DEFAULTS.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(default).alias(col))
    return out


def predict_windows(instant_df, model, class_labels, device="cpu", batch_size=512, chunk_windows=25_000, run_id="block3_long_large"):
    """Predict every window in a frame-level table. Works in chunks to keep memory down."""
    window_uids = instant_df.select("window_uid").unique().sort("window_uid")["window_uid"].to_list()
    chunks = [window_uids[i:i + chunk_windows] for i in range(0, len(window_uids), chunk_windows)]
    rows = []
    model.eval()
    for chunk in tqdm(chunks, desc="window chunks"):
        ds = WindowDataset(instant_df.filter(pl.col("window_uid").is_in(chunk)), keep_metadata=True, show_progress=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=(device == "cuda"))
        with torch.no_grad():
            for batch in loader:
                logits, embedding = model(batch["tokens"].to(device), batch["padding_mask"].to(device), return_embedding=True)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                embedding = embedding.cpu().numpy()
                n_tokens = batch["n_tokens"].numpy()
                for i in range(len(batch["window_uid"])):
                    pred_idx = int(probs[i].argmax())
                    rec = {
                        "run_id": run_id, "feature_mode": "block3",
                        "window_uid": batch["window_uid"][i], "unique_id": batch["unique_id"][i],
                        "n_valid_frames": int(n_tokens[i]),
                        "pred_idx": pred_idx, "pred_label": class_labels[pred_idx], "pred_confidence": float(probs[i, pred_idx]),
                    }
                    for col in ds.metadata_cols:
                        rec.setdefault(col, batch[col][i])
                    for j, label in enumerate(class_labels):
                        rec[f"prob_{label}"] = float(probs[i, j])
                    for j in range(embedding.shape[1]):
                        rec[f"emb_{j:02d}"] = float(embedding[i, j])
                    rows.append(rec)
        del ds, loader
        gc.collect()
    return pd.DataFrame(rows)


def add_softmax_states(df, class_labels, high_conf_threshold=0.70, mixed_top1_max=0.70, mixed_top2_min=0.20, mixed_margin_max=0.40,
                       model_unknown_top1_max=0.70, model_unknown_margin_max=0.20, model_unknown_entropy_min=0.70, model_unknown_score_min=0.55):
    """Softmax summary columns + transformer_behavior_state (thresholds in configs/pipeline.yaml)."""
    out = df.copy()
    probs = out[[f"prob_{c}" for c in class_labels]].to_numpy(dtype=float)
    rows = np.arange(len(out))
    order = np.argsort(-probs, axis=1)
    top1_idx, top2_idx = order[:, 0], order[:, 1]
    out["top1_label"] = [class_labels[i] for i in top1_idx]
    out["top2_label"] = [class_labels[i] for i in top2_idx]
    out["top1_prob"] = probs[rows, top1_idx]
    out["top2_prob"] = probs[rows, top2_idx]
    out["prob_margin"] = out["top1_prob"] - out["top2_prob"]
    out["least_confidence"] = 1.0 - out["top1_prob"]
    out["entropy"] = -(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    out["entropy_norm"] = out["entropy"] / np.log2(len(class_labels))
    out["top2_pair"] = ["_".join(sorted([a, b])) for a, b in zip(out["top1_label"], out["top2_label"])]
    out["high_conf_threshold_used"] = high_conf_threshold
    out["mixed_top1_max_used"] = mixed_top1_max
    out["mixed_top2_min_used"] = mixed_top2_min
    out["mixed_margin_max_used"] = mixed_margin_max

    out["model_unknown_score"] = (out["least_confidence"] + (1.0 - out["prob_margin"].clip(0, 1)) + out["entropy_norm"]) / 3.0
    out["is_model_unknown_like"] = (
        (out["top1_prob"] <= model_unknown_top1_max)
        | (out["prob_margin"] <= model_unknown_margin_max)
        | (out["entropy_norm"] >= model_unknown_entropy_min)
        | (out["model_unknown_score"] >= model_unknown_score_min)
    )
    out["is_high_confidence"] = out["top1_prob"] >= high_conf_threshold
    out["is_boundary_like"] = (
        (out["top1_prob"] <= mixed_top1_max) & (out["top2_prob"] >= mixed_top2_min) & (out["prob_margin"] <= mixed_margin_max)
    )
    out["is_unresolved_low_confidence"] = ~(out["is_high_confidence"] | out["is_boundary_like"])

    out["decision_type"] = "unresolved_low_confidence"
    out.loc[out["is_high_confidence"], "decision_type"] = "high_confidence_single"
    out.loc[out["is_boundary_like"], "decision_type"] = "boundary_like_pair"

    out["softmax_mixture_label"] = "unresolved_" + out["top1_label"]
    out.loc[out["is_high_confidence"], "softmax_mixture_label"] = out.loc[out["is_high_confidence"], "top1_label"]
    out.loc[out["is_boundary_like"], "softmax_mixture_label"] = out.loc[out["is_boundary_like"], "top2_pair"]
    out["transformer_behavior_state"] = out["softmax_mixture_label"].astype(str)
    return out


def add_predictions(table, behavior_df):
    """Left-join prediction columns (those not already in table) onto a polars table by window_uid."""
    table = table.with_columns(pl.col("window_uid").cast(pl.Utf8))
    join_cols = [c for c in behavior_df.columns if c == "window_uid" or c not in table.columns]
    predictions = pl.from_pandas(behavior_df[join_cols]).with_columns(pl.col("window_uid").cast(pl.Utf8))
    return table.join(predictions, on="window_uid", how="left")
