"""Step 2b: 60-frame window features for every track, in parallel (port of SPTnano.HPC).

Input:  instant_df.csv (from notebook 01)
Output: instant_df_processed.csv (adds window_uid / time_window to every frame)
        time_windowed_df_processed.csv (one row per window)

window_uid = {unique_id}_{time_window}_{frame_start}_{frame_end}. With overlap 30, each frame keeps the
window_uid of the LAST window that covers it, so most window_uids own 30 frames in the instant table.
The transformer was trained on exactly this, so do not change it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import pandas as pd
from joblib import Parallel, delayed

from .features import ParticleMetrics

WINDOW_PARAM_KEYS = [
    "window_size",
    "overlap",
    "allow_partial_window",
    "min_window_size",
    "fit_method",
    "bad_fit_strategy",
    "r2_threshold",
    "anomalous_r2_threshold",
    "use_ci",
    "ci_multiplier",
    "entropy_bins",
    "pausing_speed_threshold",
]


def window_params_from_settings(settings: dict) -> dict:
    return {key: settings["windows"][key] for key in WINDOW_PARAM_KEYS}


def split_by_track(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    unique_ids = df["unique_id"].unique()
    return [df[df["unique_id"].isin(unique_ids[i : i + chunk_size])].copy() for i in range(0, len(unique_ids), chunk_size)]


def process_chunk(chunk_df, chunk_id, frame_interval_s, window_params):
    start = time.time()
    try:
        metrics = ParticleMetrics(chunk_df, frame_interval_s)
        metrics.calculate_time_windowed_metrics(**window_params)
        print(f"chunk {chunk_id}: {chunk_df['unique_id'].nunique()} tracks in {time.time() - start:.1f}s")
        return metrics.metrics_df, metrics.get_time_windowed_df(), chunk_id
    except Exception as e:
        print(f"chunk {chunk_id} FAILED: {e}")
        return None, None, chunk_id


def compute_windows(input_file, output_dir, *, window_params: dict, frame_interval_s=0.01, n_jobs=-1, chunk_size=2000):
    """Run calculate_time_windowed_metrics over chunks of tracks and write the two output CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} points, {df['unique_id'].nunique():,} tracks from {input_file}")
    chunks = split_by_track(df, chunk_size)
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
    print(f"{len(chunks)} chunks of up to {chunk_size} tracks on {n_jobs} cores")
    print("window parameters:", window_params)

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_chunk)(chunk, i, frame_interval_s, window_params) for i, chunk in enumerate(chunks)
    )
    failed = [chunk_id for instant, windowed, chunk_id in results if instant is None]
    if failed:
        raise RuntimeError(f"Chunks failed: {failed}")

    instant_df = pd.concat([r[0] for r in results], ignore_index=True)
    windowed_df = pd.concat([r[1] for r in results], ignore_index=True)
    instant_path = os.path.join(output_dir, "instant_df_processed.csv")
    windowed_path = os.path.join(output_dir, "time_windowed_df_processed.csv")
    instant_df.to_csv(instant_path, index=False)
    windowed_df.to_csv(windowed_path, index=False)
    print(f"Saved {len(instant_df):,} frames -> {instant_path}")
    print(f"Saved {len(windowed_df):,} windows -> {windowed_path}")
    return instant_path, windowed_path
