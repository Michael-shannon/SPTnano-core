"""Step 4a: train / val / test split of the labelled windows (low-train, high-test).

1. Keep instant rows whose window has a usable certain label (not unknown / ungatedgate1).
2. Every track from the test replicate (R3) goes to test.
3. From the rest, the 12 cells (filenames) with the most tracks in each condition_comp also go to test,
   as whole cells.
4. The remaining tracks are split 90/10 into train/val, stratified by molecule, by whole track.

Works lazily on the parquet files so the full 26M-row instant table never has to be in memory.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

CONDITION_COMP_COLUMNS = ["mol", "cell", "type", "location", "geno", "group", "condition"]


def add_condition_comp(lf):
    """mol_cell_type_location_geno_group_condition, e.g. HTT_neuron_vent_GC_20H20S_HTT77_Condition_..."""
    return lf.with_columns(
        pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null("Unknown") for c in CONDITION_COMP_COLUMNS], separator="_").alias("condition_comp")
    )


def stratified_track_split(track_info, label_col, val_fraction, seed):
    """Shuffle the tracks of each label and put val_fraction of them in val."""
    rng = np.random.default_rng(seed)
    train_tracks, val_tracks = [], []
    for label in sorted(track_info[label_col].drop_nulls().unique().to_list()):
        tracks = track_info.filter(pl.col(label_col) == label)["unique_id"].to_numpy()
        rng.shuffle(tracks)
        n_val = int(len(tracks) * val_fraction)
        val_tracks.extend(str(x) for x in tracks[:n_val])
        train_tracks.extend(str(x) for x in tracks[n_val:])
    null_tracks = track_info.filter(pl.col(label_col).is_null())["unique_id"].to_numpy()
    if len(null_tracks):
        rng.shuffle(null_tracks)
        n_val = int(len(null_tracks) * val_fraction)
        val_tracks.extend(str(x) for x in null_tracks[:n_val])
        train_tracks.extend(str(x) for x in null_tracks[n_val:])
    return train_tracks, val_tracks


def pick_test_cells(base_lf, cells_per_condition):
    """The cells with the most tracks in each condition_comp (ties broken by filename)."""
    cell_summary = (
        base_lf.group_by(["filename", "condition_comp"])
        .agg(pl.col("unique_id").n_unique().alias("n_tracks"))
        .collect()
        .sort(["condition_comp", "n_tracks", "filename"], descending=[False, True, False])
        .with_columns((pl.int_range(pl.len()).over("condition_comp") + 1).alias("rank_within_condition"))
    )
    selected = cell_summary.filter(pl.col("rank_within_condition") <= cells_per_condition)
    return selected["filename"].to_list(), dict(selected.group_by("condition_comp").agg(pl.len()).iter_rows())


def write_lazy(lf, path):
    try:
        lf.sink_parquet(path)
    except Exception:
        lf.collect().write_parquet(path)


def make_splits(instant_parquet, windowed_parquet, out_dir, split_settings):
    """Write train_df / val_df / test_df.parquet and data_splits.pkl into out_dir. Returns row counts."""
    s = split_settings
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labelled_windows = (
        pl.scan_parquet(windowed_parquet)
        .filter(pl.col(s["label_col"]).is_not_null() & ~pl.col(s["label_col"]).is_in(s["exclude_labels"]))
        .select("window_uid").drop_nulls().unique().collect()
    )
    print(f"labelled windows: {labelled_windows.height:,}")

    source_lf = add_condition_comp(pl.scan_parquet(instant_parquet).join(labelled_windows.lazy(), on="window_uid", how="semi"))
    forced_lf = source_lf.filter(pl.col("replicate") == s["test_replicate"])
    base_lf = source_lf.filter(pl.col("replicate") != s["test_replicate"])

    test_cells, cells_selected_per_condition = pick_test_cells(base_lf, s["cells_per_condition"])
    print(f"test cells: {len(test_cells):,} from {len(cells_selected_per_condition):,} conditions (+ all {s['test_replicate']})")

    remaining_lf = base_lf.filter(~pl.col("filename").is_in(test_cells))
    track_info = (
        remaining_lf.group_by("unique_id").agg(pl.col(s["balance_col"]).first()).collect().sort("unique_id")
    )
    train_tracks, val_tracks = stratified_track_split(track_info, s["balance_col"], s["val_fraction"], s["seed"])
    print(f"train tracks: {len(train_tracks):,}  val tracks: {len(val_tracks):,}")

    paths = {name: out_dir / f"{name}_df.parquet" for name in ("train", "val", "test")}
    write_lazy(remaining_lf.join(pl.DataFrame({"unique_id": train_tracks}).lazy(), on="unique_id", how="semi"), paths["train"])
    write_lazy(remaining_lf.join(pl.DataFrame({"unique_id": val_tracks}).lazy(), on="unique_id", how="semi"), paths["val"])
    write_lazy(pl.concat([base_lf.filter(pl.col("filename").is_in(test_cells)), forced_lf]), paths["test"])

    counts = {}
    for name, path in paths.items():
        summary = pl.scan_parquet(path).select(pl.len().alias("rows"), pl.col("window_uid").n_unique().alias("windows")).collect()
        counts[name] = summary.row(0, named=True)
    print(counts)

    split_config = {
        **{f"{name}_path": str(path) for name, path in paths.items()},
        "splits_save_dir": str(out_dir),
        "windowed_df_path": str(windowed_parquet),
        "instant_df_path": str(instant_parquet),
        "label_col": s["label_col"],
        "split_only_certain": True,
        "excluded_split_labels": sorted(s["exclude_labels"]),
        "split_strategy": "fixed_cells",
        "balancing_features": [s["balance_col"]],
        "val_split": s["val_fraction"],
        "cells_per_condition": s["cells_per_condition"],
        "test_condition_col": "condition_comp",
        "random_seed": s["seed"],
        "force_replicate_to_test": True,
        "force_test_replicate_value": s["test_replicate"],
    }
    split_info = {
        "strategy": "fixed_cells_lowtrain_hightest",
        "test_cells": test_cells,
        "train_tracks": len(train_tracks),
        "val_tracks": len(val_tracks),
        "cells_selected_per_condition": cells_selected_per_condition,
        "cells_per_condition": s["cells_per_condition"],
    }
    with open(out_dir / "data_splits.pkl", "wb") as f:
        pickle.dump({"split_config": split_config, "split_info": split_info}, f)
    return counts


def load_splits(splits_dir):
    splits_dir = Path(splits_dir)
    return tuple(pl.read_parquet(splits_dir / f"{name}_df.parquet") for name in ("train", "val", "test"))


def check_splits(splits_dir):
    """No track may be in two splits."""
    splits_dir = Path(splits_dir)
    uids = {name: set(pl.scan_parquet(splits_dir / f"{name}_df.parquet").select("unique_id").unique().collect()["unique_id"].to_list())
            for name in ("train", "val", "test")}
    overlaps = {f"{a}/{b}": len(uids[a] & uids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    print("tracks:", {k: len(v) for k, v in uids.items()}, "overlaps:", overlaps)
    assert not any(overlaps.values()), "a track is in more than one split"
    return overlaps
