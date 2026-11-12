"""Step 6c: simple baselines, trained and scored on the same split as the transformer.

Random forest on the window scalars (the table the gates were drawn on). It is run twice:
with every scalar, and without the four gate axes, because with them it can simply redraw the gates.
Both are upper bounds rather than competitors: the labels come from these scalars.

Random forest on the block3 tokens: the same raw-motion input the transformer gets, flattened.

Gaussian HMM on per-frame motion (log step length, cos turning angle), 4 hidden states, fitted
on the training tracks without labels. Each window takes its most common hidden state, and each
hidden state is named after the gate label it overlaps most in the training windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .evaluate import CLASS_ORDER, classification_summary
from .tokens import WindowDataset

WINDOW_SCALARS = [
    "avg_msd", "diffusion_coefficient", "anomalous_exponent", "avg_speed_um_s", "avg_acceleration_um_s2",
    "avg_jerk_um_s3", "avg_normalized_curvature", "avg_angle_normalized_curvature", "persistence_length",
    "net_displacement_um", "cum_displacement_um", "straightness_index", "radius_of_gyration",
    "convex_hull_area", "directional_entropy", "speed_variability", "direction_autocorrelation",
    "kappa_turning", "kappa_absolute", "eccentricity", "turning_angle_variance", "turning_angle_skew",
    "turning_angle_kurtosis", "steplength_mean", "steplength_std", "steplength_skew", "steplength_kurtosis",
    "diffusivity_cv", "fractal_dimension", "psd_slope_speed", "self_intersections", "pausing_fraction",
]
GATE_AXES = ["avg_speed_um_s", "self_intersections", "anomalous_exponent", "cum_displacement_um"]


def split_window_tables(splits_dir, windowed_parquet, label_col="final_population_certain", labels=CLASS_ORDER):
    """Window-scalar rows for the train and test windows, keeping only the 4 behaviour labels."""
    windowed = pl.scan_parquet(windowed_parquet)
    tables = {}
    for split in ("train", "test"):
        uids = pl.scan_parquet(Path(splits_dir) / f"{split}_df.parquet").select("window_uid").unique()
        tables[split] = (
            windowed.join(uids, on="window_uid", how="semi")
            .filter(pl.col(label_col).is_in(labels))
            .unique("window_uid", keep="last")
            .collect()
        )
        print(f"{split}: {tables[split].height:,} windows")
    return tables


def scalar_matrix(df, feature_cols, fill_values=None):
    x = df.select(feature_cols).to_pandas().astype(float).replace([np.inf, -np.inf], np.nan)
    if fill_values is None:
        fill_values = x.median()
    return x.fillna(fill_values).to_numpy(), fill_values


def random_forest_benchmark(train, test, feature_cols, label_col="final_population_certain", seed=42):
    """Same forest settings as the 2026 raw-motion baseline notebook."""
    from sklearn.ensemble import RandomForestClassifier

    x_train, fill_values = scalar_matrix(train, feature_cols)
    x_test, _ = scalar_matrix(test, feature_cols, fill_values)
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(x_train, train[label_col].to_numpy())
    pred = pd.DataFrame({
        "window_uid": test["window_uid"].to_numpy(),
        label_col: test[label_col].to_numpy(),
        "pred_label": forest.predict(x_test),
    })
    importance = pd.Series(forest.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return pred, importance


def token_table(instant_parquet, windows, label_col="final_population_certain"):
    """Flattened block3 tokens (20 tokens x [sum dx, sum dy, heading]) for the labelled windows."""
    label_lookup = dict(zip(windows["window_uid"].to_list(), windows[label_col].to_list()))
    instant = pl.read_parquet(instant_parquet, columns=["window_uid", "unique_id", "frame", "x_um", "y_um", "direction_rad"])
    ds = WindowDataset(instant, label_lookup=label_lookup, str_to_idx={label: label for label in CLASS_ORDER})
    n_tokens = ds.max_seq_len // 3
    x = np.stack([s["tokens"][:n_tokens].ravel() for s in ds.samples])
    labels = np.array([s["label"] for s in ds.samples])
    window_uids = np.array([s["window_uid"] for s in ds.samples])
    return x, labels, window_uids


def token_random_forest_benchmark(splits_dir, train, test, label_col="final_population_certain", seed=42):
    from sklearn.ensemble import RandomForestClassifier

    x_train, y_train, _ = token_table(Path(splits_dir) / "train_df.parquet", train, label_col)
    x_test, y_test, test_uids = token_table(Path(splits_dir) / "test_df.parquet", test, label_col)
    forest = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    forest.fit(x_train, y_train)
    return pd.DataFrame({"window_uid": test_uids, label_col: y_test, "pred_label": forest.predict(x_test)})


def frame_motion_features(instant_parquet):
    """One row per frame: log step length and cos of the turning angle, along each track."""
    frames = (
        pl.scan_parquet(instant_parquet)
        .select("unique_id", "window_uid", "frame", "x_um", "y_um")
        .sort(["unique_id", "frame"])
        .with_columns(
            dx=pl.col("x_um").diff().over("unique_id"),
            dy=pl.col("y_um").diff().over("unique_id"),
        )
        .with_columns(heading=pl.arctan2(pl.col("dy"), pl.col("dx")))
        .with_columns(
            log_step=(pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt().add(1e-4).log(),
            cos_turn=pl.col("heading").diff().over("unique_id").cos(),
        )
        .drop_nulls(["log_step", "cos_turn"])
        .select("unique_id", "window_uid", "log_step", "cos_turn")
        .collect()
    )
    lengths = frames.group_by("unique_id", maintain_order=True).len()["len"].to_numpy()
    return frames, lengths


def hmm_benchmark(splits_dir, train_windows, label_col="final_population_certain", n_states=4, seed=42):
    from hmmlearn.hmm import GaussianHMM

    train_frames, train_lengths = frame_motion_features(Path(splits_dir) / "train_df.parquet")
    test_frames, test_lengths = frame_motion_features(Path(splits_dir) / "test_df.parquet")
    print(f"HMM frames: train {train_frames.height:,} ({len(train_lengths):,} tracks), "
          f"test {test_frames.height:,} ({len(test_lengths):,} tracks)")

    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=seed)
    hmm.fit(train_frames.select("log_step", "cos_turn").to_numpy(), train_lengths)

    def window_states(frames, lengths):
        states = hmm.predict(frames.select("log_step", "cos_turn").to_numpy(), lengths)
        return (
            frames.select("window_uid")
            .with_columns(hmm_state=pl.Series(states))
            .group_by("window_uid")
            .agg(pl.col("hmm_state").mode().first())
        )

    train_states = window_states(train_frames, train_lengths).join(
        train_windows.select("window_uid", label_col), on="window_uid", how="inner"
    )
    state_names = (
        train_states.group_by("hmm_state", label_col).len()
        .sort("len", descending=True)
        .unique("hmm_state", keep="first")
    )
    state_to_label = dict(zip(state_names["hmm_state"].to_list(), state_names[label_col].to_list()))
    print("HMM state -> label:", dict(sorted(state_to_label.items())))

    test_states = window_states(test_frames, test_lengths)
    pred = test_states.with_columns(
        pred_label=pl.col("hmm_state").replace_strict(state_to_label, default="unknown")
    )
    return pred, state_to_label, hmm


def run_benchmarks(splits_dir, windowed_parquet, out_dir, label_col="final_population_certain", seed=42):
    """Fit every baseline, score it on the held-out test windows and write benchmark_scores.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = split_window_tables(splits_dir, windowed_parquet, label_col)
    train, test = tables["train"], tables["test"]

    rows = []
    feature_sets = {
        "random_forest_all_scalars": WINDOW_SCALARS,
        "random_forest_no_gate_axes": [c for c in WINDOW_SCALARS if c not in GATE_AXES],
    }
    for name, cols in feature_sets.items():
        print(f"\n{name}: {len(cols)} features")
        pred, importance = random_forest_benchmark(train, test, cols, label_col, seed)
        scores, report = classification_summary(pred, true_col=label_col)
        rows.append({"method": name, **scores})
        importance.to_csv(out_dir / f"{name}_feature_importance.csv", header=["importance"])
        report.to_csv(out_dir / f"{name}_classification_report.csv")

    print("\nrandom_forest_block3_tokens")
    pred = token_random_forest_benchmark(splits_dir, train, test, label_col, seed)
    scores, report = classification_summary(pred, true_col=label_col)
    rows.append({"method": "random_forest_block3_tokens", **scores})
    report.to_csv(out_dir / "random_forest_block3_tokens_classification_report.csv")

    try:
        print("\ngaussian_hmm")
        pred, state_to_label, _ = hmm_benchmark(splits_dir, train, label_col, seed=seed)
        pred = pred.join(test.select("window_uid", label_col), on="window_uid", how="inner").to_pandas()
        scores, report = classification_summary(pred, true_col=label_col)
        rows.append({"method": "gaussian_hmm", **scores})
        report.to_csv(out_dir / "gaussian_hmm_classification_report.csv")
    except ImportError:
        print("hmmlearn is not installed (pip install hmmlearn) - HMM skipped")

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(out_dir / "benchmark_scores.csv", index=False)
    print("\nSaved:", out_dir / "benchmark_scores.csv")
    return scores_df
