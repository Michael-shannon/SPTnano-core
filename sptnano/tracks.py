"""Step 1b: load MATLAB tracks (*_tracked.mat), convert to microns/seconds, clean them.

The track helpers are copied unchanged from SPTnano helper_scripts. The cleaning order used for the paper:
    read_mat_file -> add_microns_and_secs (per file) -> concat -> add_unique_id
    -> filter_stubs(0.6 s) -> pathfixer(split at jumps > 1.0 um, drop pieces < 0.6 s)
"""

import os

import numpy as np
import pandas as pd
import scipy.io


def add_unique_id(df):
    df["file_id"] = pd.Categorical(df["filename"]).codes
    df["particleint"] = df["particle"].astype(int)
    df["unique_id"] = df["file_id"].astype(str) + "_" + df["particleint"].astype(str)

    return df


def add_microns_and_secs(df, pixelsize_microns, time_between_frames):
    """Adds columns to the DataFrame with positions in microns and time in seconds"""
    # space transformations
    df["x_um"] = df["x"] * pixelsize_microns
    df["y_um"] = df["y"] * pixelsize_microns

    df["frame_zeroed"] = df.groupby("particle")["frame"].transform(
        lambda x: x - x.iloc[0]
    )
    df["time_s"] = df["frame"] * time_between_frames
    df["time_s_zeroed"] = df.groupby("particle")["time_s"].transform(
        lambda x: x - x.iloc[0]
    )
    return df


def read_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    tr_data = mat["data"]["tr"][0, 0]

    # Initialize an empty list to store the combined data
    combined_data = []

    # Iterate over the cells in tr_data and concatenate them
    for cell in tr_data[0]:
        if cell.size > 0:
            combined_data.append(cell)

    # Convert the combined data into a numpy array
    combined_data = np.vstack(combined_data)

    # Create a DataFrame with the appropriate columns
    # Adjust column names based on inspection
    df = pd.DataFrame(
        combined_data,
        columns=[
            "x",
            "y",
            "frame",
            "particle",
            "column5",
            "column6",
            "column7",
            "column8",
        ],
    )

    return df


def filter_stubs(df, min_time):
    """
    Removes tracks that are shorter than 'min_time' by finding the max duration of each time_s_zeroed column and filtering on that
    Works across exposure times, because it works on converted seconds, not frames

    """
    # Calculate the duration of each track by grouping by 'particle' and using the 'time_s' column
    track_durations = df.groupby("unique_id")["time_s_zeroed"].max()
    # Identify particles with tracks longer than 0.2 seconds
    valid_particles = track_durations[track_durations >= min_time].index
    # Filter the dataframe to include only valid particles
    filtered_df = df[df["unique_id"].isin(valid_particles)]

    return filtered_df


def pathfixer(
    df,
    segment_length_threshold=0.3,
    remove_short_tracks=True,
    min_track_length_seconds=1.0,
    time_between_frames=0.1,
):
    """
    Cleans and splits tracks based on segment length thresholds. Tracks with segments exceeding
    the threshold are split, erroneous segments are removed, and resulting subtracks are
    reassigned unique IDs.

    Parameters
    - df: pd.DataFrame, input dataframe containing particle tracking data with `x_um`, `y_um`, `unique_id`, and `time_s`.
    - segment_length_threshold: float, the maximum allowable segment length (in micrometers).
    - remove_short_tracks: bool, whether to remove tracks shorter than the minimum track length after splitting.
    - min_track_length_seconds: float, the minimum track duration in seconds to be retained.
    - time_between_frames: float, time duration between frames in seconds.

    Returns
    - cleaned_df: pd.DataFrame, the cleaned and split dataframe.
    - removed_unique_ids: list, `unique_id`s that were removed due to short length or other reasons.
    - report: dict, a summary report detailing the modifications made to the dataset.

    """
    if min_track_length_seconds is None:
        raise ValueError("Please specify `min_track_length_seconds`.")

    min_track_length_frames = int(min_track_length_seconds / time_between_frames)

    original_track_count = df["unique_id"].nunique()
    original_row_count = len(df)

    # Sort by track ID and frame order
    df = df.sort_values(by=["unique_id", "frame"]).reset_index(drop=True)

    # Compute segment lengths
    df[["x_um_prev", "y_um_prev"]] = df.groupby("unique_id")[["x_um", "y_um"]].shift(1)
    df["segment_len_um"] = np.sqrt(
        (df["x_um"] - df["x_um_prev"]) ** 2 + (df["y_um"] - df["y_um_prev"]) ** 2
    ).fillna(0)

    # Assign `file_id` for consistent unique_id generation
    df["file_id"] = pd.Categorical(df["filename"]).codes

    # Initialize variables
    new_tracks = []
    split_track_count = 0
    deleted_segments_count = 0
    removed_unique_ids = set()
    next_particle_id = 0  # Using an integer instead of float to avoid formatting issues

    # Process each track individually
    for unique_id, track_data in df.groupby("unique_id"):
        current_track = []

        for _, row in track_data.iterrows():
            if row["segment_len_um"] > segment_length_threshold:
                deleted_segments_count += 1
                if current_track:
                    split_track_count += 1
                    track_df = pd.DataFrame(current_track)
                    track_df["particle"] = next_particle_id
                    track_df["file_id"] = track_data["file_id"].iloc[0]
                    track_df["unique_id"] = (
                        track_df["file_id"].astype(str) + "_" + str(next_particle_id)
                    )
                    # df['unique_id'] = df['file_id'].astype(str) + '_' + df['particle'].astype(str)
                    new_tracks.append(track_df)
                    next_particle_id += 1
                    current_track = []
            else:
                current_track.append(row)

        # Save any remaining valid track
        if current_track:
            track_df = pd.DataFrame(current_track)
            track_df["particle"] = next_particle_id
            track_df["file_id"] = track_data["file_id"].iloc[0]
            track_df["unique_id"] = (
                track_df["file_id"].astype(str) + "_" + str(next_particle_id)
            )
            new_tracks.append(track_df)
            next_particle_id += 1

    # Combine all new tracks
    cleaned_df = (
        pd.concat(new_tracks, ignore_index=True)
        if new_tracks
        else pd.DataFrame(columns=df.columns)
    )

    # Remove short tracks if enabled
    if remove_short_tracks and not cleaned_df.empty:
        cleaned_df["track_length"] = cleaned_df.groupby("unique_id")[
            "unique_id"
        ].transform("size")
        removed_unique_ids = set(
            cleaned_df.loc[
                cleaned_df["track_length"] < min_track_length_frames, "unique_id"
            ]
        )
        cleaned_df = cleaned_df[cleaned_df["track_length"] >= min_track_length_frames]
        cleaned_df.drop(columns=["track_length"], inplace=True)

    # Generate report
    final_track_count = cleaned_df["unique_id"].nunique()
    final_row_count = len(cleaned_df)

    report = {
        "original_track_count": original_track_count,
        "final_track_count": final_track_count,
        "tracks_split": split_track_count,
        "deleted_segments": deleted_segments_count,
        "original_row_count": original_row_count,
        "final_row_count": final_row_count,
    }

    return cleaned_df, removed_unique_ids, report


def load_tracked_mat_files(folder, pixel_size_um, frame_interval_s):
    """Find every *_tracked.mat under `folder`; condition = parent folder name, filename = file stem."""
    dataframes = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in sorted(filenames):
            if not name.endswith("_tracked.mat"):
                continue
            file_path = os.path.join(dirpath, name)
            df = read_mat_file(file_path)
            df["condition"] = os.path.basename(os.path.dirname(file_path))
            df["filename"] = os.path.splitext(name)[0]
            df = add_microns_and_secs(df, pixel_size_um, frame_interval_s)
            dataframes.append(df)
            print(f"  loaded {name} from {df['condition'].iloc[0]}")
    if not dataframes:
        raise FileNotFoundError(f"No *_tracked.mat files found under {folder}")
    df = pd.concat(dataframes, ignore_index=True)
    df = add_unique_id(df)
    print(f"{len(dataframes)} files, {len(df):,} points, {df['unique_id'].nunique():,} tracks")
    return df


def clean_tracks(df, min_track_time_s, max_jump_um, min_piece_time_s, frame_interval_s):
    """filter_stubs then pathfixer. Returns (cleaned_df, report)."""
    n_tracks = df["unique_id"].nunique()
    df = filter_stubs(df, min_time=min_track_time_s)
    print(f"filter_stubs({min_track_time_s} s): {n_tracks:,} -> {df['unique_id'].nunique():,} tracks")
    cleaned_df, _removed_ids, report = pathfixer(
        df,
        segment_length_threshold=max_jump_um,
        min_track_length_seconds=min_piece_time_s,
        time_between_frames=frame_interval_s,
    )
    print(f"pathfixer(split > {max_jump_um} um, keep >= {min_piece_time_s} s): {report}")
    return cleaned_df, report


def track_length_summary(df, frame_interval_s):
    lengths = df.groupby("unique_id").size()
    return pd.DataFrame({"frames": lengths.describe(), "seconds": (lengths * frame_interval_s).describe()})
