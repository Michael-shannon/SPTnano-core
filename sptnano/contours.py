"""Cell outlines (minimal-bridge contours), contour galleries, raw-movie stills and follow-track views."""

import imageio
import math
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize, is_color_like
from pathlib import Path
from scipy.interpolate import splev, splprep

from . import config
from .raw_images import (
    load_image,
    _coerce_lut_colormap,
    _as_pandas_copy,
    _single_value_from_df,
    _resolve_raw_movie_path,
    _normalize_raw_image,
    _load_raw_frame_indices,
)


def save_movie(
    viewer,
    tracks,
    feature="particle",
    save_path="movie.mov",
    steps=None,
    timer_overlay=False,
    timer_format="{time:.2f}s",
    fps=100,
):
    import numpy as np
    import pandas as pd

    # Build a complete timer mapping if timer overlay is enabled and 'time_s' exists.
    if timer_overlay and "time_s" in tracks.columns:
        min_frame = tracks["frame"].min()
        max_frame = tracks["frame"].max()
        # Since frames are floats, use np.arange with a 1.0 step.
        timer_df = pd.DataFrame({"frame": np.arange(min_frame, max_frame + 1, 1.0)})
        # Merge with available time data (dropping duplicate frames if any)
        time_data = tracks[["frame", "time_s"]].drop_duplicates()
        timer_df = timer_df.merge(time_data, on="frame", how="left")
        # Interpolate missing time_s values
        timer_df["time_s"] = timer_df["time_s"].interpolate()
    else:
        timer_df = None

    # Ensure the viewer is in 2D mode.
    viewer.dims.ndisplay = 2

    # Get all unique frames from the track data, sorted in order.
    unique_frames = np.sort(tracks["frame"].unique())
    total_frames = len(unique_frames)

    # If steps is not specified, capture every frame.
    if steps is None:
        steps = total_frames
    elif steps < total_frames:
        indices = np.linspace(0, total_frames - 1, steps, dtype=int)
        unique_frames = unique_frames[indices]
    else:
        steps = total_frames

    # Stream frames directly to disk instead of storing every screenshot in RAM.
    # Napari canvas screenshots can be large enough that collecting them first
    # will crash notebooks for long movies.
    with imageio.get_writer(save_path, fps=fps) as writer:
        for frame in unique_frames:
            viewer.dims.set_point(0, frame)
            if timer_overlay and timer_df is not None:
                update_timer_text(viewer, timer_df, frame, timer_format)
            writer.append_data(viewer.screenshot(canvas_only=True))

    print(f"Movie saved to {save_path} with {len(unique_frames)} frames at {fps} fps.")


def update_timer_text(viewer, timer_df, frame, timer_format):
    # Round the float frame to match the timer_df frame values (or convert as needed)
    frame_val = round(frame, 0)
    time_series = timer_df.loc[timer_df["frame"] == frame_val, "time_s"]
    if not time_series.empty:
        current_time = time_series.iloc[0]
    else:
        current_time = frame  # fallback
        # print(f"No time found for frame {frame}. Using frame number as fallback.")
    viewer.text_overlay.text = timer_format.format(time=current_time)


def compute_cell_edge_angle_binning(
    points, centroid, edge_angle_bins=180, edge_padding=0.05, edge_smoothing=0.1
):
    """
    Compute the cell edge using an angle–binning approach.

    The circle (0 to 2π) is divided into edge_angle_bins; for each bin, the maximum
    radial distance from the centroid is computed. Missing bins are filled by circular
    linear interpolation. The radii are then inflated by edge_padding and optionally
    smoothed in polar/radial space. Smoothing the radii rather than fitting an
    unconstrained XY spline keeps one contour point per angle, which prevents the
    outline from looping back and crossing itself.

    Parameters
      points (np.ndarray): Array of shape (N,2) of [x, y] coordinates.
      centroid (np.ndarray): [x, y] coordinates of the centroid.
      edge_angle_bins (int): Number of angular bins.
      edge_padding (float): Fractional padding (e.g., 0.05 increases each radius by 5%).
      edge_smoothing (float): Radial smoothing strength (0 means no smoothing).

    Returns
      np.ndarray: An (M,2) array with the [x, y] coordinates outlining the cell edge.

    """
    diffs = points - centroid
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angles = np.mod(angles, 2 * np.pi)
    radii = np.sqrt(np.sum(diffs**2, axis=1))

    bin_edges = np.linspace(0, 2 * np.pi, edge_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_radii = np.full_like(bin_centers, np.nan)

    for i in range(edge_angle_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if np.any(mask):
            max_radii[i] = np.max(radii[mask])

    if np.any(np.isnan(max_radii)):
        valid = ~np.isnan(max_radii)
        if np.sum(valid) < 2:
            max_radii = np.zeros_like(bin_centers)
        else:
            angles_ext = np.concatenate(
                (bin_centers[valid], np.array([bin_centers[valid][0] + 2 * np.pi]))
            )
            radii_ext = np.concatenate(
                (max_radii[valid], np.array([max_radii[valid][0]]))
            )
            max_radii = np.interp(bin_centers, angles_ext, radii_ext)

    padded_radii = max_radii * (1 + edge_padding)
    if edge_smoothing > 0 and len(padded_radii) > 2:
        # Circular Gaussian smoothing in radius/angle space preserves angular order.
        # Values < 1 are treated as a fraction of the angular sampling density.
        sigma_bins = (
            float(edge_smoothing) * edge_angle_bins / (2 * np.pi)
            if edge_smoothing < 1
            else float(edge_smoothing)
        )
        sigma_bins = max(sigma_bins, 0.25)
        radius = max(1, int(math.ceil(3 * sigma_bins)))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-(offsets**2) / (2 * sigma_bins**2))
        kernel = kernel / kernel.sum()
        padded_ext = np.pad(padded_radii, radius, mode="wrap")
        padded_radii = np.convolve(padded_ext, kernel, mode="valid")

    x_edge = centroid[0] + padded_radii * np.cos(bin_centers)
    y_edge = centroid[1] + padded_radii * np.sin(bin_centers)
    edge_points = np.vstack((x_edge, y_edge)).T

    # Repeat the first point so line plots visibly close the contour.
    return np.vstack([edge_points, edge_points[0]])


def compute_cell_edge_angle_binning_legacy_outer_envelope(
    points,
    centroid,
    edge_angle_bins=180,
    edge_padding=0.05,
    edge_smoothing=0.1,
    legacy_spline_points=200,
):
    """
    Approximate the older XY-spline contour, then keep only its outer radial envelope.

    The old XY spline could create loops. This keeps the old contour's reach by
    sampling the spline, then re-binning those points by angle and taking the
    outermost radius in each angular direction. The final contour is therefore
    ordered around the cell and cannot self-cross.
    """
    diffs = points - centroid
    angles = np.mod(np.arctan2(diffs[:, 1], diffs[:, 0]), 2 * np.pi)
    radii = np.sqrt(np.sum(diffs**2, axis=1))

    bin_edges = np.linspace(0, 2 * np.pi, edge_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_radii = np.full_like(bin_centers, np.nan)

    for i in range(edge_angle_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if np.any(mask):
            max_radii[i] = np.max(radii[mask])

    if np.any(np.isnan(max_radii)):
        valid = ~np.isnan(max_radii)
        if np.sum(valid) < 2:
            max_radii = np.zeros_like(bin_centers)
        else:
            angles_ext = np.concatenate(
                (bin_centers[valid], np.array([bin_centers[valid][0] + 2 * np.pi]))
            )
            radii_ext = np.concatenate(
                (max_radii[valid], np.array([max_radii[valid][0]]))
            )
            max_radii = np.interp(bin_centers, angles_ext, radii_ext)

    padded_radii = max_radii * (1 + edge_padding)
    old_edge = np.vstack(
        [
            centroid[0] + padded_radii * np.cos(bin_centers),
            centroid[1] + padded_radii * np.sin(bin_centers),
        ]
    )

    envelope_points = old_edge.T
    if edge_smoothing > 0:
        try:
            tck, _u = splprep(old_edge, s=edge_smoothing, per=True)
            u_fine = np.linspace(0, 1, legacy_spline_points)
            spline_points = np.array(splev(u_fine, tck)).T
            envelope_points = np.vstack([envelope_points, spline_points])
        except Exception as e:
            print("Legacy smoothing failed; using unsmoothed envelope:", e)

    return compute_cell_edge_angle_binning(
        envelope_points,
        centroid,
        edge_angle_bins=edge_angle_bins,
        edge_padding=0,
        edge_smoothing=0,
    )


def _draw_disk_on_mask(mask, center_y, center_x, radius):
    radius = int(max(0, radius))
    y0 = max(0, int(round(center_y)) - radius)
    y1 = min(mask.shape[0], int(round(center_y)) + radius + 1)
    x0 = max(0, int(round(center_x)) - radius)
    x1 = min(mask.shape[1], int(round(center_x)) + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (
        (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    )


def _draw_line_on_mask(mask, point_a_xy, point_b_xy, width):
    y0, x0 = int(round(point_a_xy[1])), int(round(point_a_xy[0]))
    y1, x1 = int(round(point_b_xy[1])), int(round(point_b_xy[0]))
    n_steps = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    ys = np.linspace(y0, y1, n_steps)
    xs = np.linspace(x0, x1, n_steps)
    for y, x in zip(ys, xs):
        _draw_disk_on_mask(mask, y, x, width)


def _line_hits_forbidden(point_a_xy, point_b_xy, forbidden_mask):
    if forbidden_mask is None:
        return False
    y0, x0 = int(round(point_a_xy[1])), int(round(point_a_xy[0]))
    y1, x1 = int(round(point_b_xy[1])), int(round(point_b_xy[0]))
    n_steps = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    ys = np.clip(np.round(np.linspace(y0, y1, n_steps)).astype(int), 0, forbidden_mask.shape[0] - 1)
    xs = np.clip(np.round(np.linspace(x0, x1, n_steps)).astype(int), 0, forbidden_mask.shape[1] - 1)
    return bool(np.any(forbidden_mask[ys, xs]))


def _polygon_to_mask(contour_xy, shape, min_xy):
    from skimage.draw import polygon

    contour_xy = np.asarray(contour_xy, dtype=float) - min_xy
    rr, cc = polygon(contour_xy[:, 1], contour_xy[:, 0], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _smooth_closed_contour(contour, smoothing):
    if smoothing <= 0 or len(contour) < 5:
        return contour
    sigma = float(smoothing) * len(contour) / (2 * np.pi) if smoothing < 1 else float(smoothing)
    sigma = max(sigma, 0.25)
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    smoothed = []
    for dim in range(contour.shape[1]):
        padded = np.pad(contour[:, dim], radius, mode="wrap")
        smoothed.append(np.convolve(padded, kernel, mode="valid"))
    return np.column_stack(smoothed)


def compute_cell_edge_minimal_bridge(
    points,
    *,
    point_radius=2.0,
    bridge_width=1.5,
    mask_padding=12,
    edge_padding=0.0,
    edge_smoothing=0.0,
    forbidden_contours=None,
    enforce_point_containment=True,
):
    """
    Build a single contour from point blobs joined by narrow shortest bridges.

    This is intended for cells/growth cones where the point footprint has sparse
    or separated regions. Unlike radial/outer-envelope methods, it does not
    inflate across empty angular sectors; separated components are connected by
    narrow line bridges between their closest points before extracting the
    contour from the resulting mask.
    """
    from scipy.ndimage import binary_fill_holes
    from scipy.spatial import cKDTree
    from skimage.measure import find_contours, label

    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError("At least three points are required to compute a contour")

    point_radius = float(point_radius)
    bridge_width = float(bridge_width)
    pad = int(max(mask_padding, point_radius * 4, bridge_width * 4, 4))

    min_xy = np.floor(points.min(axis=0)).astype(int) - pad
    max_xy = np.ceil(points.max(axis=0)).astype(int) + pad
    for forbidden in forbidden_contours or []:
        forbidden = np.asarray(forbidden, dtype=float)
        if len(forbidden):
            min_xy = np.minimum(min_xy, np.floor(forbidden.min(axis=0)).astype(int) - pad)
            max_xy = np.maximum(max_xy, np.ceil(forbidden.max(axis=0)).astype(int) + pad)
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    shifted = points - min_xy

    mask = np.zeros((height, width), dtype=bool)
    forbidden_mask = None
    if forbidden_contours:
        forbidden_mask = np.zeros_like(mask, dtype=bool)
        for forbidden in forbidden_contours:
            forbidden_mask |= _polygon_to_mask(forbidden, mask.shape, min_xy)

    for x, y in shifted:
        _draw_disk_on_mask(mask, y, x, point_radius)
    if forbidden_mask is not None:
        mask &= ~forbidden_mask

    for _ in range(100):
        labels = label(mask, connectivity=2)
        component_ids = [component_id for component_id in np.unique(labels) if component_id != 0]
        if len(component_ids) <= 1:
            break

        point_labels = labels[
            np.clip(np.round(shifted[:, 1]).astype(int), 0, height - 1),
            np.clip(np.round(shifted[:, 0]).astype(int), 0, width - 1),
        ]
        component_points = {
            component_id: shifted[point_labels == component_id]
            for component_id in component_ids
        }
        component_points = {
            component_id: pts
            for component_id, pts in component_points.items()
            if len(pts) > 0
        }
        if len(component_points) <= 1:
            break

        best_pair = None
        best_distance = np.inf
        component_items = list(component_points.items())
        for idx, (component_a, points_a) in enumerate(component_items[:-1]):
            tree = cKDTree(points_a)
            for component_b, points_b in component_items[idx + 1:]:
                distances, indices = tree.query(points_b, k=1)
                for candidate_idx in np.argsort(distances):
                    candidate_idx = int(candidate_idx)
                    point_a = points_a[int(indices[candidate_idx])]
                    point_b = points_b[candidate_idx]
                    if _line_hits_forbidden(point_a, point_b, forbidden_mask):
                        continue
                    distance = float(distances[candidate_idx])
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (point_a, point_b, component_a, component_b)
                    break
        if best_pair is None:
            break
        _draw_line_on_mask(mask, best_pair[0], best_pair[1], bridge_width)
        if forbidden_mask is not None:
            mask &= ~forbidden_mask

    if edge_padding > 0:
        from scipy.ndimage import binary_dilation

        iterations = max(1, int(round(edge_padding * max(width, height))))
        mask = binary_dilation(mask, iterations=iterations)

    mask = binary_fill_holes(mask)
    if forbidden_mask is not None:
        mask &= ~forbidden_mask
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("Could not extract a contour from the minimal-bridge mask")

    contour_rc = max(contours, key=len)
    unsmoothed_contour_xy = np.column_stack([contour_rc[:, 1], contour_rc[:, 0]]) + min_xy
    contour_xy = _smooth_closed_contour(unsmoothed_contour_xy, edge_smoothing)
    if enforce_point_containment:
        contains = mpltPath.Path(contour_xy).contains_points(points, radius=1.0)
        if not np.all(contains):
            contour_xy = unsmoothed_contour_xy
            contains = mpltPath.Path(contour_xy).contains_points(points, radius=1.0)
        dilation_count = 0
        while not np.all(contains) and dilation_count < 20:
            from scipy.ndimage import binary_dilation

            mask = binary_dilation(mask, iterations=1)
            if forbidden_mask is not None:
                mask &= ~forbidden_mask
            contours = find_contours(mask.astype(float), level=0.5)
            if not contours:
                break
            contour_rc = max(contours, key=len)
            contour_xy = np.column_stack([contour_rc[:, 1], contour_rc[:, 0]]) + min_xy
            contains = mpltPath.Path(contour_xy).contains_points(points, radius=1.0)
            dilation_count += 1
    return np.vstack([contour_xy, contour_xy[0]])


def _restore_dataframe_type(df_pandas, template_df):
    """Return a dataframe with the same backend as the input template."""
    if isinstance(template_df, pl.DataFrame):
        return pl.from_pandas(df_pandas)
    return df_pandas


def find_track_rich_frame_range(
    single_cell_df,
    *,
    target_tracks=30,
    window_frames=None,
    window_seconds=15,
    step_frames=None,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    start_frame=None,
    min_track_points=2,
    track_id_col="unique_id",
    frame_col="frame",
):
    """
    Find the first time window containing at least ``target_tracks`` usable tracks.

    This is a frame-based version of the older auto-selection logic used by
    ``plot_tracks_polars_static``. It deliberately returns only a frame range so
    track selection can stay outside the rendering functions.
    """
    df = _as_pandas_copy(single_cell_df)
    for col in [track_id_col, frame_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found")

    if window_frames is None:
        window_frames = int(round(window_seconds / frame_interval))
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")

    min_frame = int(df[frame_col].min())
    max_frame = int(df[frame_col].max())
    if start_frame is None:
        start_frame = min_frame
    else:
        start_frame = max(min_frame, int(start_frame))

    if step_frames is None:
        step_frames = max(1, int(round(1.0 / frame_interval)))

    best = None
    for current_start in range(start_frame, max_frame + 1, step_frames):
        current_end = min(max_frame, current_start + window_frames - 1)
        window = df[(df[frame_col] >= current_start) & (df[frame_col] <= current_end)]
        counts = window.groupby(track_id_col).size()
        usable = counts[counts >= min_track_points]
        n_tracks = int(len(usable))

        if best is None or n_tracks > best["n_tracks"]:
            best = {
                "frame_start": current_start,
                "frame_end": current_end,
                "n_tracks": n_tracks,
            }

        if n_tracks >= target_tracks:
            return {
                "frame_start": current_start,
                "frame_end": current_end,
                "n_tracks": n_tracks,
                "target_tracks": target_tracks,
                "target_met": True,
            }

    if best is None:
        raise ValueError("Could not find a valid frame window")

    best["target_tracks"] = target_tracks
    best["target_met"] = False
    return best


def find_population_rich_frame_range(
    single_cell_df,
    *,
    populations=None,
    population_col="final_population",
    min_tracks_per_population=1,
    window_frames=None,
    window_seconds=15,
    step_frames=None,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    start_frame=None,
    min_track_points=2,
    track_id_col="unique_id",
    frame_col="frame",
):
    """
    Find a frame window containing examples from every requested population.

    A population is considered present when at least ``min_tracks_per_population``
    unique tracks have at least ``min_track_points`` rows in that population within
    the candidate window. This makes it useful for choosing one movie segment that
    contains all behaviour types at some point in the clip.
    """
    df = _as_pandas_copy(single_cell_df)
    for col in [track_id_col, frame_col, population_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found")

    if populations is None:
        populations = sorted(pd.Series(df[population_col]).dropna().unique())
    else:
        populations = list(populations)
    if len(populations) == 0:
        raise ValueError("No populations were provided or found")

    if isinstance(min_tracks_per_population, dict):
        min_by_population = {
            pop: int(min_tracks_per_population.get(pop, 1)) for pop in populations
        }
    else:
        min_by_population = {
            pop: int(min_tracks_per_population) for pop in populations
        }

    if window_frames is None:
        window_frames = int(round(window_seconds / frame_interval))
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")

    min_frame = int(df[frame_col].min())
    max_frame = int(df[frame_col].max())
    if start_frame is None:
        start_frame = min_frame
    else:
        start_frame = max(min_frame, int(start_frame))

    if step_frames is None:
        step_frames = max(1, int(round(1.0 / frame_interval)))

    def summarize_window(window_df):
        counts_by_population = {}
        for pop in populations:
            pop_window = window_df[window_df[population_col] == pop]
            if pop_window.empty:
                counts_by_population[pop] = 0
                continue
            track_counts = pop_window.groupby(track_id_col).size()
            usable = track_counts[track_counts >= min_track_points]
            counts_by_population[pop] = int(len(usable))
        missing = [
            pop
            for pop in populations
            if counts_by_population[pop] < min_by_population[pop]
        ]
        return counts_by_population, missing

    best = None
    for current_start in range(start_frame, max_frame + 1, step_frames):
        current_end = min(max_frame, current_start + window_frames - 1)
        window = df[(df[frame_col] >= current_start) & (df[frame_col] <= current_end)]
        counts_by_population, missing = summarize_window(window)
        populations_present = len(populations) - len(missing)
        total_population_tracks = int(sum(counts_by_population.values()))

        candidate = {
            "frame_start": current_start,
            "frame_end": current_end,
            "counts_by_population": counts_by_population,
            "missing_populations": missing,
            "populations_present": populations_present,
            "total_population_tracks": total_population_tracks,
        }

        if best is None:
            best = candidate
        else:
            best_score = (
                best["populations_present"],
                best["total_population_tracks"],
            )
            candidate_score = (populations_present, total_population_tracks)
            if candidate_score > best_score:
                best = candidate

        if len(missing) == 0:
            candidate["target_met"] = True
            candidate["populations"] = populations
            candidate["min_tracks_per_population"] = min_by_population
            return candidate

    if best is None:
        raise ValueError("Could not find a valid frame window")

    best["target_met"] = False
    best["populations"] = populations
    best["min_tracks_per_population"] = min_by_population
    return best


def filter_tracks_for_raw_overlay(
    single_cell_df,
    *,
    frame_range=None,
    track_id_col="unique_id",
    frame_col="frame",
    min_track_points=2,
    min_track_length_frames=None,
    target_tracks=30,
    selection_method="longest",
    include_track_ids=None,
    population_col=None,
    populations=None,
    tracks_per_population=None,
    random_state=42,
):
    """
    Select the track subset to overlay, without doing any rendering.

    The returned dataframe keeps all rows for the selected track IDs, so the same
    selection can be used consistently for quick QC, napari movies, and stills.
    """
    df = _as_pandas_copy(single_cell_df)
    for col in [track_id_col, frame_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found")

    if population_col is not None and populations is not None:
        if population_col not in df.columns:
            raise ValueError(f"Column {population_col!r} not found")
        populations = list(populations)
        df = df[df[population_col].isin(populations)].copy()

    if include_track_ids is not None:
        selected_ids = list(include_track_ids)
    else:
        if frame_range is None:
            window_df = df
        else:
            frame_start, frame_end = frame_range
            window_df = df[
                (df[frame_col] >= int(frame_start)) & (df[frame_col] <= int(frame_end))
            ].copy()

        if window_df.empty:
            raise ValueError("No tracks are present in the requested frame range")

        summary = (
            window_df.groupby(track_id_col)[frame_col]
            .agg(["count", "min", "max"])
            .reset_index()
        )
        summary["track_length_frames"] = summary["max"] - summary["min"] + 1
        summary = summary[summary["count"] >= min_track_points]

        if min_track_length_frames is not None:
            summary = summary[
                summary["track_length_frames"] >= int(min_track_length_frames)
            ]

        if summary.empty:
            raise ValueError("No tracks passed the selection filters")

        if selection_method == "balanced_by_population":
            if population_col is None:
                raise ValueError(
                    "population_col is required for selection_method='balanced_by_population'"
                )
            if population_col not in window_df.columns:
                raise ValueError(f"Column {population_col!r} not found")
            if populations is None:
                populations = sorted(
                    pd.Series(window_df[population_col]).dropna().unique()
                )
            else:
                populations = list(populations)
            if len(populations) == 0:
                raise ValueError("No populations available for balanced selection")

            if tracks_per_population is None:
                if target_tracks is None:
                    tracks_per_population = 1
                else:
                    tracks_per_population = max(1, int(target_tracks) // len(populations))
            tracks_per_population = int(tracks_per_population)

            selected_ids = []
            selected_by_population = {}
            for pop in populations:
                pop_window = window_df[window_df[population_col] == pop]
                pop_summary = (
                    pop_window.groupby(track_id_col)[frame_col]
                    .agg(["count", "min", "max"])
                    .reset_index()
                )
                if pop_summary.empty:
                    selected_by_population[pop] = []
                    continue
                pop_summary["track_length_frames"] = (
                    pop_summary["max"] - pop_summary["min"] + 1
                )
                pop_summary = pop_summary[pop_summary["count"] >= min_track_points]
                if min_track_length_frames is not None:
                    pop_summary = pop_summary[
                        pop_summary["track_length_frames"]
                        >= int(min_track_length_frames)
                    ]
                pop_summary = pop_summary.sort_values(
                    ["track_length_frames", "count"], ascending=False
                )
                pop_ids = pop_summary[track_id_col].head(tracks_per_population).tolist()
                selected_by_population[pop] = pop_ids
                for track_id in pop_ids:
                    if track_id not in selected_ids:
                        selected_ids.append(track_id)

            if len(selected_ids) == 0:
                raise ValueError("Balanced population selection returned no tracks")

        elif selection_method == "longest":
            summary = summary.sort_values(
                ["track_length_frames", "count"], ascending=False
            )
        elif selection_method == "most_points":
            summary = summary.sort_values(["count", "track_length_frames"], ascending=False)
        elif selection_method == "random":
            summary = summary.sample(
                frac=1, random_state=random_state
            ).reset_index(drop=True)
        else:
            raise ValueError(
                "selection_method must be 'longest', 'most_points', 'random', "
                "or 'balanced_by_population'"
            )

        if selection_method != "balanced_by_population":
            if target_tracks is not None:
                summary = summary.head(int(target_tracks))
            selected_ids = summary[track_id_col].tolist()

    selected = df[df[track_id_col].isin(selected_ids)].copy()
    if selected.empty:
        raise ValueError("Track selection returned no rows")

    return {
        "tracks_df": _restore_dataframe_type(selected, single_cell_df),
        "selected_track_ids": selected_ids,
        "n_tracks": len(selected_ids),
        "frame_range": frame_range,
    }


def choose_single_cell_for_display(
    source_df,
    *,
    mode="most_tracks",
    gallery_lookup_df=None,
    gallery_number=1,
    manual_filename=None,
    population_col="final_population",
    target_population=None,
    random_seed=42,
    group_col="filename",
    track_id_col="unique_id",
):
    """
    Select one cell/movie for display workflows.

    Modes are ``gallery_number``, ``manual``, ``random``, ``most_tracks``, and
    ``most_tracks_in_population``.
    """
    candidate_source = source_df
    if mode == "most_tracks_in_population":
        if population_col not in candidate_source.columns:
            raise ValueError(f"Column {population_col!r} not found")
        if target_population is None:
            raise ValueError("Set target_population for most_tracks_in_population")
        if isinstance(candidate_source, pl.DataFrame):
            candidate_source = candidate_source.filter(
                pl.col(population_col) == target_population
            )
        else:
            candidate_source = candidate_source[
                candidate_source[population_col] == target_population
            ]

    selected_key = None
    if mode == "gallery_number":
        if gallery_lookup_df is None:
            raise ValueError("Run the gallery first or use a non-gallery selection mode")
        lookup_pdf = _as_pandas_copy(gallery_lookup_df)
        selected_key = lookup_pdf.loc[
            lookup_pdf["gallery_number"] == gallery_number,
            group_col,
        ].iloc[0]
    elif mode == "manual":
        if manual_filename is None:
            raise ValueError("Set manual_filename when mode='manual'")
        selected_key = manual_filename
    else:
        if isinstance(candidate_source, pl.DataFrame):
            cell_track_counts = (
                candidate_source.group_by(group_col)
                .agg(
                    [
                        pl.col(track_id_col).n_unique().alias("n_tracks"),
                        pl.len().alias("n_datapoints"),
                    ]
                )
                .sort(["n_tracks", "n_datapoints"], descending=True)
            )
            if mode == "random":
                filenames = cell_track_counts.select(group_col).to_series().to_list()
                import random

                random.seed(random_seed)
                selected_key = random.choice(filenames)
            elif mode in {"most_tracks", "most_tracks_in_population"}:
                selected_key = cell_track_counts.select(group_col).to_series()[0]
            else:
                raise ValueError("Unknown display cell selection mode")
        else:
            df = _as_pandas_copy(candidate_source)
            cell_track_counts = (
                df.groupby(group_col)
                .agg(
                    n_tracks=(track_id_col, "nunique"),
                    n_datapoints=(group_col, "size"),
                )
                .reset_index()
                .sort_values(["n_tracks", "n_datapoints"], ascending=False)
            )
            if mode == "random":
                selected_key = cell_track_counts[group_col].sample(
                    n=1, random_state=random_seed
                ).iloc[0]
            elif mode in {"most_tracks", "most_tracks_in_population"}:
                selected_key = cell_track_counts[group_col].iloc[0]
            else:
                raise ValueError("Unknown display cell selection mode")

    if isinstance(source_df, pl.DataFrame):
        if group_col not in source_df.columns:
            raise ValueError(f"Column {group_col!r} not found")
        selected_cell_df = source_df.filter(pl.col(group_col) == selected_key)
    else:
        df = _as_pandas_copy(source_df)
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col!r} not found")
        selected_cell_df = df[df[group_col] == selected_key].copy()
    selected_pdf = _as_pandas_copy(selected_cell_df)
    selected_key = selected_pdf[group_col].dropna().unique()[0]
    return selected_key, selected_cell_df


def choose_frame_range_for_display(
    single_cell_df,
    *,
    mode="all",
    manual_frame_range=None,
    target_tracks=30,
    window_seconds=15,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    start_frame=None,
    min_track_points=10,
    population_col="final_population",
    populations=None,
    min_tracks_per_population=1,
    track_id_col="unique_id",
    frame_col="frame",
):
    """Choose a frame range for display workflows."""
    if mode == "all":
        df = _as_pandas_copy(single_cell_df)
        frame_range = (int(df[frame_col].min()), int(df[frame_col].max()))
        return frame_range, {"mode": "all", "target_met": None}
    if mode == "manual":
        if manual_frame_range is None:
            raise ValueError("Set manual_frame_range when mode='manual'")
        return manual_frame_range, {"mode": "manual", "target_met": None}
    if mode == "track_rich":
        info = find_track_rich_frame_range(
            single_cell_df,
            target_tracks=target_tracks,
            window_seconds=window_seconds,
            frame_interval=frame_interval,
            start_frame=start_frame,
            min_track_points=min_track_points,
            track_id_col=track_id_col,
            frame_col=frame_col,
        )
        return (info["frame_start"], info["frame_end"]), info
    if mode == "population_rich":
        info = find_population_rich_frame_range(
            single_cell_df,
            populations=populations,
            population_col=population_col,
            min_tracks_per_population=min_tracks_per_population,
            window_seconds=window_seconds,
            frame_interval=frame_interval,
            start_frame=start_frame,
            min_track_points=min_track_points,
            track_id_col=track_id_col,
            frame_col=frame_col,
        )
        return (info["frame_start"], info["frame_end"]), info
    raise ValueError("Unknown display frame range mode")


def select_tracks_for_gallery_cells(
    source_df,
    *,
    frame_range=None,
    selection_method="longest",
    target_tracks_per_cell=20,
    min_track_points=20,
    min_track_length_frames=None,
    population_col=None,
    populations=None,
    tracks_per_population=None,
    random_state=42,
    group_col="filename",
    track_id_col="unique_id",
):
    """Select track subsets independently for every cell in a gallery source df."""
    selected_cells = []
    selection_rows = []
    filenames = _as_pandas_copy(source_df)[group_col].dropna().unique().tolist()
    for filename in filenames:
        if isinstance(source_df, pl.DataFrame):
            cell_df = source_df.filter(pl.col(group_col) == filename)
        else:
            cell_df = source_df[source_df[group_col] == filename].copy()
        try:
            selection = filter_tracks_for_raw_overlay(
                cell_df,
                frame_range=frame_range,
                selection_method=selection_method,
                target_tracks=target_tracks_per_cell,
                min_track_points=min_track_points,
                min_track_length_frames=min_track_length_frames,
                population_col=population_col,
                populations=populations,
                tracks_per_population=tracks_per_population,
                random_state=random_state,
                track_id_col=track_id_col,
            )
        except ValueError:
            continue
        tracks_df = selection["tracks_df"]
        if len(_as_pandas_copy(tracks_df)) == 0:
            continue
        selected_cells.append(tracks_df)
        selection_rows.append(
            {
                group_col: filename,
                "n_selected_tracks": selection["n_tracks"],
                "selected_track_ids": selection["selected_track_ids"],
            }
        )
    if not selected_cells:
        raise ValueError("No gallery tracks were selected. Loosen the gallery filters.")
    if isinstance(source_df, pl.DataFrame):
        return pl.concat(selected_cells, how="vertical"), pl.DataFrame(selection_rows)
    return pd.concat(selected_cells, ignore_index=True), pd.DataFrame(selection_rows)


def with_cell_key(
    df,
    *,
    folder_col="foldername",
    filename_col="filename",
    key_col="cell_key",
    separator="||",
):
    """Return a dataframe with a stable unique key made from folder + filename."""
    if isinstance(df, pl.DataFrame):
        for col in [folder_col, filename_col]:
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found")
        return df.with_columns(
            (
                pl.col(folder_col).cast(pl.Utf8)
                + separator
                + pl.col(filename_col).cast(pl.Utf8)
            ).alias(key_col)
        )

    out = _as_pandas_copy(df)
    for col in [folder_col, filename_col]:
        if col not in out.columns:
            raise ValueError(f"Column {col!r} not found")
    out[key_col] = out[folder_col].astype(str) + separator + out[filename_col].astype(str)
    return out


def compute_fixed_cell_contour(
    single_cell_df,
    *,
    coordinate_units="pixels",
    x_col=None,
    y_col=None,
    contour_method="radial",
    edge_angle_bins=180,
    edge_padding=0.05,
    edge_smoothing=0.1,
    minimal_bridge_point_radius=2.0,
    minimal_bridge_width=1.5,
    minimal_bridge_mask_padding=12,
):
    """
    Compute one fixed averaged cell contour from all coordinates in one cell.
    """
    df = _as_pandas_copy(single_cell_df)
    if x_col is None:
        x_col = "x" if coordinate_units == "pixels" else "x_um"
    if y_col is None:
        y_col = "y" if coordinate_units == "pixels" else "y_um"
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found")

    points = df[[x_col, y_col]].dropna().to_numpy()
    if len(points) < 3:
        raise ValueError("At least three points are required to compute a contour")

    centroid = points.mean(axis=0)
    if contour_method == "radial":
        return compute_cell_edge_angle_binning(
            points,
            centroid,
            edge_angle_bins=edge_angle_bins,
            edge_padding=edge_padding,
            edge_smoothing=edge_smoothing,
        )
    if contour_method == "legacy_outer_envelope":
        return compute_cell_edge_angle_binning_legacy_outer_envelope(
            points,
            centroid,
            edge_angle_bins=edge_angle_bins,
            edge_padding=edge_padding,
            edge_smoothing=edge_smoothing,
        )
    if contour_method == "minimal_bridge":
        return compute_cell_edge_minimal_bridge(
            points,
            point_radius=minimal_bridge_point_radius,
            bridge_width=minimal_bridge_width,
            mask_padding=minimal_bridge_mask_padding,
            edge_padding=edge_padding,
            edge_smoothing=edge_smoothing,
        )
    raise ValueError(
        "contour_method must be 'radial', 'legacy_outer_envelope', or 'minimal_bridge'"
    )


def _polygon_area_perimeter(points):
    """Return polygon area and perimeter for an ordered Nx2 contour."""
    if points is None or len(points) < 3:
        return np.nan, np.nan
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    diffs = np.diff(np.vstack([points, points[0]]), axis=0)
    perimeter = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
    return float(area), perimeter


def _contour_from_stored_row(contour_source):
    """Extract an Nx2 contour from a one-row dataframe with contour_x/contour_y."""
    df = _as_pandas_copy(contour_source)
    if "contour_x" not in df.columns or "contour_y" not in df.columns or df.empty:
        return None

    x_values = df["contour_x"].iloc[0]
    y_values = df["contour_y"].iloc[0]
    if x_values is None or y_values is None:
        return None

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Stored contour_x and contour_y must be 1D arrays of equal length")
    return np.column_stack([x, y])


def get_contour_array(contour_source, *, coordinate_units="pixels", contour_method="radial"):
    """
    Return an Nx2 contour from either a stored contour dataframe or a track dataframe.
    """
    stored = _contour_from_stored_row(contour_source)
    if stored is not None:
        return stored
    return compute_fixed_cell_contour(
        contour_source,
        coordinate_units=coordinate_units,
        contour_method=contour_method,
    )


def compute_cell_contours_dataframe(
    tracks_df,
    *,
    group_col="filename",
    coordinate_units="pixels",
    x_col=None,
    y_col=None,
    metadata_cols=None,
    selection_method=None,
    n_cells=None,
    track_id_col="unique_id",
    random_state=42,
    contour_method="radial",
    edge_angle_bins=180,
    edge_padding=0.05,
    edge_smoothing=0.1,
    minimal_bridge_point_radius=2.0,
    minimal_bridge_width=1.5,
    minimal_bridge_mask_padding=12,
    on_error="raise",
    show_progress=False,
    return_backend="polars",
):
    """
    Compute fixed contours for cells and store them one row per cell.

    The contour itself is stored as list columns, ``contour_x`` and ``contour_y``.
    This keeps the table compact and easy to save as Parquet while preserving
    numeric arrays for later geometry measurements.
    """
    if x_col is None:
        x_col = "x" if coordinate_units == "pixels" else "x_um"
    if y_col is None:
        y_col = "y" if coordinate_units == "pixels" else "y_um"

    if isinstance(tracks_df, pl.DataFrame):
        if group_col not in tracks_df.columns:
            raise ValueError(f"Column {group_col!r} not found")
        for col in [x_col, y_col]:
            if col not in tracks_df.columns:
                raise ValueError(f"Column {col!r} not found")

        agg_exprs = [pl.len().alias("n_datapoints")]
        if track_id_col in tracks_df.columns:
            agg_exprs.append(pl.col(track_id_col).n_unique().alias("n_tracks"))
        else:
            agg_exprs.append(pl.len().alias("n_tracks"))
        summary_pl = tracks_df.group_by(group_col).agg(agg_exprs)

        if selection_method == "most_tracks":
            summary_pl = summary_pl.sort(["n_tracks", "n_datapoints"], descending=True)
        elif selection_method == "most_points":
            summary_pl = summary_pl.sort(["n_datapoints", "n_tracks"], descending=True)
        elif selection_method == "random":
            summary_pl = summary_pl.sample(fraction=1.0, shuffle=True, seed=random_state)
        elif selection_method is not None:
            raise ValueError(
                "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
            )
        if n_cells is not None:
            summary_pl = summary_pl.head(int(n_cells))

        metadata_cols = list(metadata_cols or [])
        metadata_cols = [
            col for col in metadata_cols if col in tracks_df.columns and col != group_col
        ]
        selected_cells = summary_pl[group_col].to_list()
        selected_cell_set = set(selected_cells)
        selected_tracks_df = tracks_df.filter(pl.col(group_col).is_in(selected_cells))

        grouped_cells = selected_tracks_df.group_by(group_col, maintain_order=True)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                grouped_cells = tqdm(
                    grouped_cells,
                    total=len(selected_cells),
                    desc="Computing cell contours",
                    unit="cell",
                )
            except Exception:
                pass

        rows = []
        for group_key, sub_pl in grouped_cells:
            cell_id = group_key[0] if isinstance(group_key, tuple) else group_key
            if cell_id not in selected_cell_set:
                continue
            sub = sub_pl.to_pandas()
            points = sub[[x_col, y_col]].dropna().to_numpy()
            if len(points) < 3:
                continue
            try:
                contour = compute_fixed_cell_contour(
                    sub,
                    coordinate_units=coordinate_units,
                    x_col=x_col,
                    y_col=y_col,
                    contour_method=contour_method,
                    edge_angle_bins=edge_angle_bins,
                    edge_padding=edge_padding,
                    edge_smoothing=edge_smoothing,
                    minimal_bridge_point_radius=minimal_bridge_point_radius,
                    minimal_bridge_width=minimal_bridge_width,
                    minimal_bridge_mask_padding=minimal_bridge_mask_padding,
                )
            except Exception as exc:
                if on_error == "raise":
                    raise ValueError(
                        f"Failed to compute contour for {group_col}={cell_id!r}"
                    ) from exc
                if on_error != "record":
                    raise ValueError("on_error must be 'raise' or 'record'") from exc
                centroid = points.mean(axis=0)
                row = {
                    group_col: cell_id,
                    "contour_id": 1,
                    "coordinate_units": coordinate_units,
                    "x_col": x_col,
                    "y_col": y_col,
                    "contour_x": [],
                    "contour_y": [],
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "area": np.nan,
                    "perimeter": np.nan,
                    "circumference": np.nan,
                    "bbox_min_x": np.nan,
                    "bbox_max_x": np.nan,
                    "bbox_min_y": np.nan,
                    "bbox_max_y": np.nan,
                    "n_contour_points": 0,
                    "n_source_points": int(len(points)),
                    "n_tracks": int(sub[track_id_col].nunique())
                    if track_id_col in sub.columns
                    else np.nan,
                    "edge_angle_bins": int(edge_angle_bins),
                    "edge_padding": float(edge_padding),
                    "edge_smoothing": float(edge_smoothing),
                    "minimal_bridge_point_radius": float(minimal_bridge_point_radius),
                    "minimal_bridge_width": float(minimal_bridge_width),
                    "minimal_bridge_mask_padding": int(minimal_bridge_mask_padding),
                    "contour_method": contour_method,
                    "contour_status": "failed",
                    "contour_error": str(exc),
                }
                for col in metadata_cols:
                    row[col] = _single_value_from_df(sub, col)
                rows.append(row)
                continue
            area, perimeter = _polygon_area_perimeter(contour)
            centroid = points.mean(axis=0)
            row = {
                group_col: cell_id,
                "contour_id": 1,
                "coordinate_units": coordinate_units,
                "x_col": x_col,
                "y_col": y_col,
                "contour_x": contour[:, 0].astype(float).tolist(),
                "contour_y": contour[:, 1].astype(float).tolist(),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "area": area,
                "perimeter": perimeter,
                "circumference": perimeter,
                "bbox_min_x": float(np.min(contour[:, 0])),
                "bbox_max_x": float(np.max(contour[:, 0])),
                "bbox_min_y": float(np.min(contour[:, 1])),
                "bbox_max_y": float(np.max(contour[:, 1])),
                "n_contour_points": int(len(contour)),
                "n_source_points": int(len(points)),
                "n_tracks": int(sub[track_id_col].nunique())
                if track_id_col in sub.columns
                else np.nan,
                "edge_angle_bins": int(edge_angle_bins),
                "edge_padding": float(edge_padding),
                "edge_smoothing": float(edge_smoothing),
                "minimal_bridge_point_radius": float(minimal_bridge_point_radius),
                "minimal_bridge_width": float(minimal_bridge_width),
                "minimal_bridge_mask_padding": int(minimal_bridge_mask_padding),
                "contour_method": contour_method,
                "contour_status": "ok",
                "contour_error": None,
            }
            for col in metadata_cols:
                row[col] = _single_value_from_df(sub, col)
            rows.append(row)

        result = pd.DataFrame(rows)
        if return_backend == "pandas":
            return result
        if return_backend == "polars":
            return pl.from_pandas(result)
        raise ValueError("return_backend must be 'polars' or 'pandas'")

    df = _as_pandas_copy(tracks_df)
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not found")

    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found")

    metadata_cols = list(metadata_cols or [])
    metadata_cols = [col for col in metadata_cols if col in df.columns and col != group_col]

    cell_summary = (
        df.groupby(group_col)
        .agg(
            n_datapoints=(x_col, "size"),
            n_tracks=(track_id_col, "nunique") if track_id_col in df.columns else (x_col, "size"),
        )
        .reset_index()
    )

    if selection_method == "most_tracks":
        cell_summary = cell_summary.sort_values(
            ["n_tracks", "n_datapoints"], ascending=False
        )
    elif selection_method == "most_points":
        cell_summary = cell_summary.sort_values(
            ["n_datapoints", "n_tracks"], ascending=False
        )
    elif selection_method == "random":
        cell_summary = cell_summary.sample(
            frac=1, random_state=random_state
        ).reset_index(drop=True)
    elif selection_method is not None:
        raise ValueError(
            "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
        )

    if n_cells is not None:
        cell_summary = cell_summary.head(int(n_cells))

    cell_ids = cell_summary[group_col].tolist()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            cell_ids = tqdm(
                cell_ids,
                total=len(cell_ids),
                desc="Computing cell contours",
                unit="cell",
            )
        except Exception:
            pass

    rows = []
    for cell_id in cell_ids:
        sub = df[df[group_col] == cell_id].copy()
        points = sub[[x_col, y_col]].dropna().to_numpy()
        if len(points) < 3:
            continue

        try:
            contour = compute_fixed_cell_contour(
                sub,
                coordinate_units=coordinate_units,
                x_col=x_col,
                y_col=y_col,
                contour_method=contour_method,
                edge_angle_bins=edge_angle_bins,
                edge_padding=edge_padding,
                edge_smoothing=edge_smoothing,
                minimal_bridge_point_radius=minimal_bridge_point_radius,
                minimal_bridge_width=minimal_bridge_width,
                minimal_bridge_mask_padding=minimal_bridge_mask_padding,
            )
        except Exception as exc:
            if on_error == "raise":
                raise ValueError(
                    f"Failed to compute contour for {group_col}={cell_id!r}"
                ) from exc
            if on_error != "record":
                raise ValueError("on_error must be 'raise' or 'record'") from exc
            centroid = points.mean(axis=0)
            row = {
                group_col: cell_id,
                "contour_id": 1,
                "coordinate_units": coordinate_units,
                "x_col": x_col,
                "y_col": y_col,
                "contour_x": [],
                "contour_y": [],
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "area": np.nan,
                "perimeter": np.nan,
                "circumference": np.nan,
                "bbox_min_x": np.nan,
                "bbox_max_x": np.nan,
                "bbox_min_y": np.nan,
                "bbox_max_y": np.nan,
                "n_contour_points": 0,
                "n_source_points": int(len(points)),
                "n_tracks": int(sub[track_id_col].nunique())
                if track_id_col in sub.columns
                else np.nan,
                "edge_angle_bins": int(edge_angle_bins),
                "edge_padding": float(edge_padding),
                "edge_smoothing": float(edge_smoothing),
                "minimal_bridge_point_radius": float(minimal_bridge_point_radius),
                "minimal_bridge_width": float(minimal_bridge_width),
                "minimal_bridge_mask_padding": int(minimal_bridge_mask_padding),
                "contour_method": contour_method,
                "contour_status": "failed",
                "contour_error": str(exc),
            }
            for col in metadata_cols:
                row[col] = _single_value_from_df(sub, col)
            rows.append(row)
            continue
        area, perimeter = _polygon_area_perimeter(contour)
        centroid = points.mean(axis=0)
        row = {
            group_col: cell_id,
            "contour_id": 1,
            "coordinate_units": coordinate_units,
            "x_col": x_col,
            "y_col": y_col,
            "contour_x": contour[:, 0].astype(float).tolist(),
            "contour_y": contour[:, 1].astype(float).tolist(),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "area": area,
            "perimeter": perimeter,
            "circumference": perimeter,
            "bbox_min_x": float(np.min(contour[:, 0])),
            "bbox_max_x": float(np.max(contour[:, 0])),
            "bbox_min_y": float(np.min(contour[:, 1])),
            "bbox_max_y": float(np.max(contour[:, 1])),
            "n_contour_points": int(len(contour)),
            "n_source_points": int(len(points)),
            "n_tracks": int(sub[track_id_col].nunique()) if track_id_col in sub.columns else np.nan,
            "edge_angle_bins": int(edge_angle_bins),
            "edge_padding": float(edge_padding),
            "edge_smoothing": float(edge_smoothing),
            "minimal_bridge_point_radius": float(minimal_bridge_point_radius),
            "minimal_bridge_width": float(minimal_bridge_width),
            "minimal_bridge_mask_padding": int(minimal_bridge_mask_padding),
            "contour_method": contour_method,
            "contour_status": "ok",
            "contour_error": None,
        }
        for col in metadata_cols:
            row[col] = _single_value_from_df(sub, col)
        rows.append(row)

    result = pd.DataFrame(rows)
    if return_backend == "pandas":
        return result
    if return_backend == "polars":
        return pl.from_pandas(result)
    raise ValueError("return_backend must be 'polars' or 'pandas'")


def add_contour_measurements_by_filename(
    df,
    contours_df,
    *,
    group_col="filename",
    contour_id_col="contour_id",
    area_col="area",
    perimeter_col="perimeter",
    circumference_col="circumference",
    prefix="contour_",
):
    """
    Join one-contour-per-filename measurements onto a track dataframe.

    This is intended for the settled single-contour workflow. If ``contours_df``
    contains multiple contour rows for a filename, the first contour_id after
    sorting is used to avoid duplicating track rows.
    """
    output_is_polars = isinstance(df, pl.DataFrame)

    if isinstance(contours_df, pl.DataFrame):
        contour_columns = contours_df.columns
        required = [group_col, contour_id_col, area_col, perimeter_col]
        for col in required:
            if col not in contour_columns:
                raise ValueError(f"Column {col!r} not found in contours_df")
        circumference_expr = (
            pl.col(circumference_col)
            if circumference_col in contour_columns
            else pl.col(perimeter_col)
        )
        mapping = (
            contours_df.sort([group_col, contour_id_col])
            .group_by(group_col, maintain_order=True)
            .agg(
                [
                    pl.col(contour_id_col).first().alias(f"{prefix}id"),
                    pl.col(area_col).first().alias(f"{prefix}area"),
                    pl.col(perimeter_col).first().alias(f"{prefix}perimeter"),
                    circumference_expr.first().alias(f"{prefix}circumference"),
                ]
            )
        )
        if output_is_polars:
            return df.join(mapping, on=group_col, how="left")
        return _as_pandas_copy(df).merge(mapping.to_pandas(), on=group_col, how="left")

    contour_pdf = _as_pandas_copy(contours_df)
    required = [group_col, contour_id_col, area_col, perimeter_col]
    for col in required:
        if col not in contour_pdf.columns:
            raise ValueError(f"Column {col!r} not found in contours_df")
    if circumference_col not in contour_pdf.columns:
        contour_pdf[circumference_col] = contour_pdf[perimeter_col]
    mapping_pdf = (
        contour_pdf.sort_values([group_col, contour_id_col])
        .groupby(group_col, as_index=False)
        .first()
    )
    mapping_pdf = mapping_pdf[
        [group_col, contour_id_col, area_col, perimeter_col, circumference_col]
    ].rename(
        columns={
            contour_id_col: f"{prefix}id",
            area_col: f"{prefix}area",
            perimeter_col: f"{prefix}perimeter",
            circumference_col: f"{prefix}circumference",
        }
    )
    if output_is_polars:
        return df.join(pl.from_pandas(mapping_pdf), on=group_col, how="left")
    return _as_pandas_copy(df).merge(mapping_pdf, on=group_col, how="left")


def plot_segmented_contour_gallery(
    tracks_df,
    contours_df,
    *,
    n_cells=16,
    selection_method="most_tracks",
    group_col="filename",
    track_id_col="unique_id",
    contour_id_col="contour_id",
    track_ids_col="track_ids",
    coordinate_units="pixels",
    x_col=None,
    y_col=None,
    n_cols=4,
    figsize_per_cell=(2, 2),
    contour_colors=None,
    contour_colorway="tab10",
    contour_line_width=1.0,
    contour_line_style="--",
    show_contour_ids=True,
    contour_id_fontsize=8,
    show_panel_numbers=True,
    panel_number_start=1,
    show_tracks=True,
    track_color_mode="by_contour",
    track_color="0.55",
    track_colorway="tab20",
    track_line_width=0.7,
    track_alpha=0.85,
    max_tracks_per_cell=80,
    frame_range=None,
    frame_col="frame",
    show_raw_projection=False,
    raw_projection="max",
    raw_projection_frame_range=None,
    raw_max_frames=250,
    raw_master_dir=None,
    raw_contrast_limits=None,
    invert_raw=False,
    raw_cmap="gray",
    raw_lut=None,
    raw_alpha=1.0,
    title_col=None,
    save_path=None,
    dpi=150,
    show_plot=True,
):
    """
    Gallery for segmented contour results.

    Unlike ``plot_cell_contour_gallery``, this groups multiple contour rows with
    the same filename into a single panel and labels each ``contour_id``.
    """
    contour_pdf = _as_pandas_copy(contours_df)
    if contour_pdf.empty:
        raise ValueError("contours_df is empty")
    for col in [group_col, contour_id_col, "contour_x", "contour_y"]:
        if col not in contour_pdf.columns:
            raise ValueError(f"Column {col!r} not found in contours_df")

    if isinstance(tracks_df, pl.DataFrame):
        if group_col not in tracks_df.columns:
            raise ValueError(f"Column {group_col!r} not found in tracks_df")
        summary_pl = tracks_df.group_by(group_col).agg(
            [
                pl.len().alias("n_datapoints"),
                pl.col(track_id_col).n_unique().alias("n_tracks")
                if track_id_col in tracks_df.columns
                else pl.len().alias("n_tracks"),
            ]
        )
        if selection_method == "most_tracks":
            summary_pl = summary_pl.sort(["n_tracks", "n_datapoints"], descending=True)
        elif selection_method == "most_points":
            summary_pl = summary_pl.sort(["n_datapoints", "n_tracks"], descending=True)
        elif selection_method == "random":
            summary_pl = summary_pl.sample(fraction=1.0, shuffle=True, seed=42)
        elif selection_method is not None:
            raise ValueError(
                "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
            )
        available_cells = set(contour_pdf[group_col].dropna().unique())
        selected_cells = [
            cell for cell in summary_pl[group_col].to_list() if cell in available_cells
        ]
        if n_cells is not None:
            selected_cells = selected_cells[: int(n_cells)]
        selected_tracks = tracks_df.filter(pl.col(group_col).is_in(selected_cells))
        if frame_range is not None and frame_col in selected_tracks.columns:
            frame_start, frame_end = map(int, frame_range)
            selected_tracks = selected_tracks.filter(
                (pl.col(frame_col) >= frame_start) & (pl.col(frame_col) <= frame_end)
            )
        if (
            max_tracks_per_cell is not None
            and track_id_col in selected_tracks.columns
            and show_tracks
        ):
            track_counts = selected_tracks.group_by([group_col, track_id_col]).agg(
                pl.len().alias("_n_track_points")
            )
            top_tracks = (
                track_counts.with_columns(
                    pl.col("_n_track_points")
                    .rank("ordinal", descending=True)
                    .over(group_col)
                    .alias("_track_rank")
                )
                .filter(pl.col("_track_rank") <= int(max_tracks_per_cell))
                .select([group_col, track_id_col])
            )
            selected_tracks = selected_tracks.join(
                top_tracks, on=[group_col, track_id_col], how="inner"
            )
        selected_track_df = selected_tracks.to_pandas()
    else:
        track_pdf = _as_pandas_copy(tracks_df)
        summary = (
            track_pdf.groupby(group_col)
            .agg(
                n_datapoints=(group_col, "size"),
                n_tracks=(track_id_col, "nunique")
                if track_id_col in track_pdf.columns
                else (group_col, "size"),
            )
            .reset_index()
        )
        if selection_method == "most_tracks":
            summary = summary.sort_values(["n_tracks", "n_datapoints"], ascending=False)
        elif selection_method == "most_points":
            summary = summary.sort_values(["n_datapoints", "n_tracks"], ascending=False)
        elif selection_method == "random":
            summary = summary.sample(frac=1, random_state=42)
        elif selection_method is not None:
            raise ValueError(
                "selection_method must be None, 'most_tracks', 'most_points', or 'random'"
            )
        available_cells = set(contour_pdf[group_col].dropna().unique())
        selected_cells = [
            cell for cell in summary[group_col].tolist() if cell in available_cells
        ]
        if n_cells is not None:
            selected_cells = selected_cells[: int(n_cells)]
        selected_track_df = track_pdf[track_pdf[group_col].isin(selected_cells)].copy()

    contour_pdf = contour_pdf[contour_pdf[group_col].isin(selected_cells)].copy()
    if contour_pdf.empty:
        raise ValueError("No segmented contours available for selected cells")

    n = len(selected_cells)
    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
        dpi=dpi,
    )
    for ax in axes.ravel():
        ax.axis("off")

    if contour_colors is None:
        contour_ids = list(pd.unique(contour_pdf[contour_id_col]))
        cmap = plt.get_cmap(contour_colorway, max(len(contour_ids), 1))
        contour_colors = {contour_id: cmap(i) for i, contour_id in enumerate(contour_ids)}

    lookup_rows = []
    for idx, cell_id in enumerate(selected_cells):
        ax = axes.ravel()[idx]
        cell_contours = contour_pdf[contour_pdf[group_col] == cell_id].copy()
        cell_tracks = selected_track_df[selected_track_df[group_col] == cell_id].copy()

        panel_number = int(panel_number_start) + idx
        foldername = _single_value_from_df(cell_tracks, "foldername", default="")
        filename_value = _single_value_from_df(cell_tracks, "filename", default=cell_id)
        lookup_rows.append(
            {
                "gallery_number": panel_number,
                group_col: cell_id,
                "filename": filename_value,
                "foldername": foldername,
                "n_contours": int(cell_contours[contour_id_col].nunique()),
            }
        )

        if show_raw_projection and not cell_tracks.empty:
            if raw_projection_frame_range is not None:
                raw_frame_start, raw_frame_end = map(int, raw_projection_frame_range)
            elif frame_range is not None:
                raw_frame_start, raw_frame_end = map(int, frame_range)
            elif frame_col in cell_tracks.columns:
                raw_frame_start = int(cell_tracks[frame_col].min())
                raw_frame_end = int(cell_tracks[frame_col].max())
            else:
                raw_frame_start = raw_frame_end = None
            if raw_frame_start is not None and raw_frame_end is not None:
                raw_movie_path = _resolve_raw_movie_path(
                    cell_tracks, master_dir=raw_master_dir
                )
                raw_frame_indices = np.arange(raw_frame_start, raw_frame_end + 1)
                if raw_max_frames is not None and len(raw_frame_indices) > int(raw_max_frames):
                    raw_frame_indices = np.linspace(
                        raw_frame_start,
                        raw_frame_end,
                        int(raw_max_frames),
                        dtype=int,
                    )
                raw_stack = _load_raw_frame_indices(raw_movie_path, raw_frame_indices)
                if raw_stack.ndim == 2:
                    raw_image = raw_stack
                elif raw_projection == "max":
                    raw_image = np.nanmax(raw_stack, axis=0)
                elif raw_projection == "sum":
                    raw_image = np.nansum(raw_stack, axis=0)
                elif raw_projection == "mean":
                    raw_image = np.nanmean(raw_stack, axis=0)
                else:
                    raise ValueError("raw_projection must be 'max', 'sum', or 'mean'")
                ax.imshow(
                    _normalize_raw_image(
                        raw_image,
                        contrast_limits=raw_contrast_limits,
                        invert_image=invert_raw,
                    ),
                    cmap=_coerce_lut_colormap(
                        raw_lut, target="matplotlib", fallback=raw_cmap
                    ),
                    origin="upper",
                    alpha=raw_alpha,
                    zorder=0,
                )

        if show_tracks and not cell_tracks.empty:
            x_col = (
                x_col
                or ("x" if coordinate_units == "pixels" and "x" in cell_tracks.columns else "x_um")
            )
            y_col = (
                y_col
                or ("y" if coordinate_units == "pixels" and "y" in cell_tracks.columns else "y_um")
            )
            if x_col not in cell_tracks.columns or y_col not in cell_tracks.columns:
                raise ValueError(f"Columns {x_col!r} and/or {y_col!r} not found in tracks_df")
            if {x_col, y_col, track_id_col}.issubset(cell_tracks.columns):
                track_to_contour = {}
                if track_ids_col in cell_contours.columns:
                    for _, contour_row in cell_contours.iterrows():
                        for track_id in contour_row[track_ids_col]:
                            track_to_contour[track_id] = contour_row[contour_id_col]
                cell_tracks["_gallery_contour_id"] = cell_tracks[track_id_col].map(
                    track_to_contour
                )
                if track_color_mode == "by_contour":
                    color_source = cell_tracks.dropna(subset=["_gallery_contour_id"])
                    track_colors = (
                        _track_color_map(
                            color_source,
                            track_id_col,
                            "_gallery_contour_id",
                            contour_colorway,
                        )
                        if not color_source.empty
                        else {}
                    )
                elif track_color_mode == "by_track":
                    track_colors = _track_color_map(
                        cell_tracks, track_id_col, track_id_col, track_colorway
                    )
                elif track_color_mode == "single":
                    track_colors = None
                else:
                    raise ValueError(
                        "track_color_mode must be 'by_contour', 'by_track', or 'single'"
                    )
                sort_col = frame_col if frame_col in cell_tracks.columns else x_col
                segments = []
                colors = []
                for track_id, track in cell_tracks.sort_values(sort_col).groupby(track_id_col):
                    if len(track) < 2:
                        continue
                    segments.append(track[[x_col, y_col]].to_numpy())
                    colors.append(
                        track_colors.get(track_id, track_color)
                        if track_colors is not None
                        else track_color
                    )
                if segments:
                    ax.add_collection(
                        LineCollection(
                            segments,
                            colors=colors,
                            linewidths=track_line_width,
                            alpha=track_alpha,
                            zorder=1,
                        )
                    )

        for _, contour_row in cell_contours.iterrows():
            contour = np.column_stack(
                [
                    np.asarray(contour_row["contour_x"], dtype=float),
                    np.asarray(contour_row["contour_y"], dtype=float),
                ]
            )
            contour_id = contour_row[contour_id_col]
            color = (
                contour_colors.get(contour_id, "yellow")
                if isinstance(contour_colors, dict)
                else None
            )
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                lw=contour_line_width,
                linestyle=contour_line_style,
                zorder=2,
            )
            if show_contour_ids:
                label_x = contour_row.get("centroid_x", np.nanmean(contour[:, 0]))
                label_y = contour_row.get("centroid_y", np.nanmean(contour[:, 1]))
                ax.text(
                    label_x,
                    label_y,
                    str(contour_id),
                    color=color,
                    fontsize=contour_id_fontsize,
                    weight="bold",
                    ha="center",
                    va="center",
                    zorder=3,
                )

        ax.set_aspect("equal", adjustable="box")
        if coordinate_units == "pixels":
            ax.invert_yaxis()
        ax.axis("off")
        if show_panel_numbers:
            ax.set_title(str(panel_number), fontsize=8)
        elif title_col is not None and title_col in cell_contours.columns:
            ax.set_title(str(_single_value_from_df(cell_contours, title_col)), fontsize=8)
        else:
            ax.set_title(str(cell_id), fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "contours_df": contour_pdf,
        "selected_cells": selected_cells,
        "cell_lookup_df": pd.DataFrame(lookup_rows),
    }


def make_raw_track_napari_movie(
    single_cell_df,
    tracks_to_plot_df,
    *,
    frame_range,
    master_dir=None,
    contour_df=None,
    raw_movie_path=None,
    track_id_col="unique_id",
    frame_col="frame",
    x_col="x",
    y_col="y",
    color_by="unique_id",
    tail_length=60,
    tail_width=2,
    contour_color="cyan",
    contour_width=2,
    raw_contrast_limits=None,
    invert_raw=False,
    raw_colormap="gray",
    raw_lut=None,
    raw_frame_offset=0,
    spatial_transform="identity",
    save_path=None,
    fps=100,
    steps=None,
    run_napari=True,
    close_after_save=False,
    background="dark",
    show_time_label=True,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    time_label_position=(10, 10),
    time_label_size=14,
    time_label_color="white",
    show_scale_bar=True,
    scale_bar_length_um=5.0,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    scale_bar_position="lower_right",
    scale_bar_margin_px=12,
    scale_bar_line_width=4,
    scale_bar_color="white",
    scale_bar_label=None,
    scale_bar_label_size=12,
):
    """
    Show/save a napari movie of raw cell frames, selected tracks, and a fixed contour.
    """
    cell_pdf = _as_pandas_copy(single_cell_df)
    tracks = _as_pandas_copy(tracks_to_plot_df)
    for col in [track_id_col, frame_col, x_col, y_col]:
        if col not in tracks.columns:
            raise ValueError(f"Column {col!r} not found in tracks_to_plot_df")

    frame_start, frame_end = map(int, frame_range)
    tracks = tracks[
        (tracks[frame_col] >= frame_start) & (tracks[frame_col] <= frame_end)
    ].copy()
    if tracks.empty:
        raise ValueError("No selected tracks are present in frame_range")

    raw_movie_path = raw_movie_path or _resolve_raw_movie_path(cell_pdf, master_dir=master_dir)
    raw_frame_start = frame_start + int(raw_frame_offset)
    raw_frame_end = frame_end + int(raw_frame_offset)
    if raw_frame_start < 0:
        raise ValueError("raw_frame_offset makes the first requested raw frame negative")
    image = _load_raw_frame_range(raw_movie_path, raw_frame_start, raw_frame_end)
    if invert_raw:
        image = np.nanmax(image) - image

    track_ids = list(pd.unique(tracks[track_id_col]))
    track_id_map = {track_id: i for i, track_id in enumerate(track_ids)}
    tracks["_napari_track_id"] = tracks[track_id_col].map(track_id_map)
    tracks["_relative_frame"] = tracks[frame_col].astype(int) - frame_start
    height, width = image.shape[-2:]
    transformed_x, transformed_y = _transform_xy_for_raw_alignment(
        tracks[x_col].to_numpy(),
        tracks[y_col].to_numpy(),
        width,
        height,
        spatial_transform,
    )
    tracks["_display_x"] = transformed_x
    tracks["_display_y"] = transformed_y
    tracks_new_df = tracks[["_napari_track_id", "_relative_frame", "_display_y", "_display_x"]].copy()
    tracks_new_df.columns = ["track_id", "frame", "y", "x"]

    features_dict = {track_id_col: tracks[track_id_col].to_numpy()}
    color_feature = track_id_col
    if color_by in tracks.columns:
        color_feature = color_by
        if pd.api.types.is_numeric_dtype(tracks[color_by]):
            features_dict[color_by] = tracks[color_by].to_numpy()
        else:
            categories = {v: i for i, v in enumerate(pd.unique(tracks[color_by]))}
            mapped_col = f"{color_by}_numeric"
            tracks[mapped_col] = tracks[color_by].map(categories)
            features_dict[mapped_col] = tracks[mapped_col].to_numpy()
            color_feature = mapped_col

    import napari

    viewer = napari.Viewer()
    viewer.theme = "light" if background == "light" else "dark"
    viewer.dims.axis_labels = ("time", "y", "x")
    viewer.add_image(
        image,
        name="Raw movie",
        contrast_limits=raw_contrast_limits,
        colormap=_coerce_lut_colormap(raw_lut, target="napari", fallback=raw_colormap),
    )

    viewer.add_tracks(
        tracks_new_df.to_numpy(),
        features=features_dict,
        name="Selected tracks",
        color_by=color_feature,
        tail_length=tail_length,
        tail_width=tail_width,
    )

    contour_source = contour_df if contour_df is not None else single_cell_df
    contour = get_contour_array(contour_source, coordinate_units="pixels")
    contour_x, contour_y = _transform_xy_for_raw_alignment(
        contour[:, 0],
        contour[:, 1],
        width,
        height,
        spatial_transform,
    )
    contour_yx = np.column_stack([contour_y, contour_x])
    viewer.add_shapes(
        [contour_yx],
        shape_type="polygon",
        name="Fixed cell contour",
        edge_color=contour_color,
        face_color="transparent",
        edge_width=contour_width,
    )

    if show_scale_bar and scale_bar_length_um is not None and pixel_size_um:
        scale_bar_length_px = float(scale_bar_length_um) / float(pixel_size_um)
        margin = float(scale_bar_margin_px)
        if scale_bar_position == "lower_left":
            x0 = margin
            x1 = x0 + scale_bar_length_px
            y = height - margin
        elif scale_bar_position == "upper_left":
            x0 = margin
            x1 = x0 + scale_bar_length_px
            y = margin
        elif scale_bar_position == "upper_right":
            x1 = width - margin
            x0 = x1 - scale_bar_length_px
            y = margin
        else:
            x1 = width - margin
            x0 = x1 - scale_bar_length_px
            y = height - margin
        viewer.add_shapes(
            [np.array([[y, x0], [y, x1]], dtype=float)],
            shape_type="line",
            name="Scale bar",
            edge_color=scale_bar_color,
            edge_width=scale_bar_line_width,
        )
        label = scale_bar_label or f"{scale_bar_length_um:g} um"
        viewer.add_points(
            np.array([[y - 10, (x0 + x1) / 2]], dtype=float),
            name="Scale bar label",
            size=0.1,
            face_color=scale_bar_color,
            opacity=1.0,
            text={
                "string": [label],
                "size": scale_bar_label_size,
                "color": scale_bar_color,
                "anchor": "center",
            },
        )

    if show_time_label:
        relative_frames = np.arange(frame_end - frame_start + 1, dtype=int)
        time_points = np.column_stack(
            [
                relative_frames,
                np.full_like(relative_frames, float(time_label_position[0]), dtype=float),
                np.full_like(relative_frames, float(time_label_position[1]), dtype=float),
            ]
        )
        time_labels = [
            f"{(frame_start + int(frame)) * float(frame_interval):.2f} s"
            for frame in relative_frames
        ]
        viewer.add_points(
            time_points,
            name="Time label",
            size=0.1,
            face_color=time_label_color,
            opacity=1.0,
            text={
                "string": time_labels,
                "size": time_label_size,
                "color": time_label_color,
                "anchor": "upper_left",
            },
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_movie(
            viewer,
            tracks_new_df,
            save_path=save_path,
            steps=steps,
            fps=fps,
        )
        if close_after_save and not run_napari:
            viewer.close()

    if run_napari:
        napari.run()

    return {
        "viewer": viewer,
        "raw_movie_path": raw_movie_path,
        "frame_range": (frame_start, frame_end),
        "raw_frame_range": (raw_frame_start, raw_frame_end),
        "spatial_transform": spatial_transform,
        "n_tracks": len(track_ids),
        "contour": contour,
    }


def _track_color_map(df, track_id_col, color_by, colorway):
    track_ids = list(pd.unique(df[track_id_col]))
    if color_by in df.columns and color_by != track_id_col:
        values = df.groupby(track_id_col)[color_by].first()
    else:
        values = pd.Series(track_ids, index=track_ids)

    if pd.api.types.is_numeric_dtype(values):
        vmin, vmax = float(values.min()), float(values.max())
        norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
        cmap = ListedColormap(colorway) if isinstance(colorway, (list, tuple)) else plt.get_cmap(colorway)
        return {track_id: cmap(norm(values.loc[track_id])) for track_id in track_ids}

    categories = list(pd.unique(values))
    if isinstance(colorway, (list, tuple)):
        cmap = ListedColormap(colorway)
    else:
        cmap = plt.get_cmap(colorway, max(len(categories), 1))
    category_colors = {cat: cmap(i) for i, cat in enumerate(categories)}
    return {track_id: category_colors[values.loc[track_id]] for track_id in track_ids}


def _load_raw_frame_range(raw_movie_path, frame_start, frame_end):
    """
    Load only the requested contiguous frame range for TIFF stacks when possible.
    """
    return _load_raw_frame_indices(
        raw_movie_path,
        np.arange(int(frame_start), int(frame_end) + 1),
    )


def save_raw_track_timelapse_stills(
    single_cell_df,
    tracks_to_plot_df,
    *,
    frame_range,
    output_dir,
    master_dir=None,
    contour_df=None,
    raw_movie_path=None,
    raw_mode="final_frame",
    still_step_frames=60,
    track_history="cumulative",
    track_id_col="unique_id",
    frame_col="frame",
    x_col="x",
    y_col="y",
    track_color_mode="by_track",
    track_color="yellow",
    color_by="unique_id",
    colorway="tab20",
    contour_color="cyan",
    contour_line_width=1.0,
    track_line_width=0.8,
    head_size=8,
    raw_cmap="gray",
    raw_lut=None,
    raw_frame_offset=0,
    spatial_transform="identity",
    contrast_limits=None,
    invert_image=False,
    figsize=(3, 3),
    dpi=200,
    output_format="png",
    output_prefix=None,
    show_time_label=True,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    show_scale_bar=True,
    scale_bar_length_um=5.0,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    scale_bar_position="lower_right",
    scale_bar_margin_px=12,
    scale_bar_line_width=3,
    scale_bar_color="white",
    scale_bar_label=None,
    scale_bar_fontsize=8,
):
    """
    Save publication-style PNG stills from raw cell frames and selected tracks.

    ``raw_mode='sum_window'`` sums the raw frames in each output window. 
    ``raw_mode='final_frame'`` uses only the last raw frame of each output window.
    Tracks can be cumulative from the start of ``frame_range`` or limited to the
    current window via ``track_history='window'``.
    """
    if raw_mode not in {"final_frame", "sum_window"}:
        raise ValueError("raw_mode must be 'final_frame' or 'sum_window'")
    if track_history not in {"cumulative", "window"}:
        raise ValueError("track_history must be 'cumulative' or 'window'")
    output_format = str(output_format).lower().lstrip(".")
    if output_format not in {"png", "svg"}:
        raise ValueError("output_format must be 'png' or 'svg'")
    if track_color_mode not in {"single", "by_track", "by_column"}:
        raise ValueError("track_color_mode must be 'single', 'by_track', or 'by_column'")

    tracks = _as_pandas_copy(tracks_to_plot_df)
    for col in [track_id_col, frame_col, x_col, y_col]:
        if col not in tracks.columns:
            raise ValueError(f"Column {col!r} not found in tracks_to_plot_df")

    frame_start, frame_end = map(int, frame_range)
    raw_movie_path = raw_movie_path or _resolve_raw_movie_path(
        single_cell_df, master_dir=master_dir
    )
    image = load_image(raw_movie_path)
    height, width = image.shape[-2:]

    os.makedirs(output_dir, exist_ok=True)
    if output_prefix is None:
        filename = _single_value_from_df(_as_pandas_copy(single_cell_df), "filename", "cell")
        output_prefix = str(filename).replace("_tracked", "")

    if still_step_frames is None:
        windows = [(frame, frame) for frame in range(frame_start, frame_end + 1)]
    else:
        still_step_frames = int(still_step_frames)
        if still_step_frames <= 0:
            raise ValueError("still_step_frames must be positive or None")
        windows = []
        current = frame_start
        while current <= frame_end:
            window_end = min(frame_end, current + still_step_frames - 1)
            windows.append((current, window_end))
            current = window_end + 1

    tracks = tracks[
        (tracks[frame_col] >= frame_start) & (tracks[frame_col] <= frame_end)
    ].copy()
    if tracks.empty:
        raise ValueError("No selected tracks are present in frame_range")

    if track_color_mode == "single":
        color_map = {track_id: track_color for track_id in pd.unique(tracks[track_id_col])}
    elif track_color_mode == "by_column":
        if color_by not in tracks.columns:
            raise ValueError(f"Column {color_by!r} not found for track_color_mode='by_column'")
        color_map = _track_color_map(tracks, track_id_col, color_by, colorway)
    else:
        color_map = _track_color_map(tracks, track_id_col, track_id_col, colorway)
    contour_source = contour_df if contour_df is not None else single_cell_df
    contour = get_contour_array(contour_source, coordinate_units="pixels")

    output_paths = []
    for i, (window_start, window_end) in enumerate(windows):
        if raw_mode == "sum_window":
            raw_start = window_start + int(raw_frame_offset)
            raw_end = window_end + int(raw_frame_offset)
            if raw_start < 0:
                raise ValueError("raw_frame_offset makes a requested raw frame negative")
            raw = image[raw_start : raw_end + 1].sum(axis=0)
        else:
            raw_index = window_end + int(raw_frame_offset)
            if raw_index < 0:
                raise ValueError("raw_frame_offset makes a requested raw frame negative")
            raw = image[raw_index]
        raw = _normalize_raw_image(raw, contrast_limits=contrast_limits, invert_image=invert_image)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(
            raw,
            cmap=_coerce_lut_colormap(raw_lut, target="matplotlib", fallback=raw_cmap),
            origin="upper",
        )
        contour_x, contour_y = _transform_xy_for_raw_alignment(
            contour[:, 0],
            contour[:, 1],
            width,
            height,
            spatial_transform,
        )
        ax.plot(
            contour_x,
            contour_y,
            color=contour_color,
            lw=contour_line_width,
            alpha=0.95,
        )

        if track_history == "cumulative":
            plot_start = frame_start
        else:
            plot_start = window_start
        visible_tracks = tracks[
            (tracks[frame_col] >= plot_start) & (tracks[frame_col] <= window_end)
        ]

        for track_id, track in visible_tracks.sort_values(frame_col).groupby(track_id_col):
            if len(track) >= 2:
                tx, ty = _transform_xy_for_raw_alignment(
                    track[x_col].to_numpy(),
                    track[y_col].to_numpy(),
                    width,
                    height,
                    spatial_transform,
                )
                ax.plot(
                    tx,
                    ty,
                    color=color_map.get(track_id, "yellow"),
                    lw=track_line_width,
                    alpha=0.9,
                )
            current = track[track[frame_col] <= window_end].tail(1)
            if not current.empty:
                cx, cy = _transform_xy_for_raw_alignment(
                    current[x_col].to_numpy(),
                    current[y_col].to_numpy(),
                    width,
                    height,
                    spatial_transform,
                )
                ax.scatter(
                    cx,
                    cy,
                    color=color_map.get(track_id, "yellow"),
                    s=head_size,
                    edgecolors="none",
                    alpha=1.0,
                )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if show_time_label:
            t0 = window_start * frame_interval
            t1 = window_end * frame_interval
            ax.text(
                0.02,
                0.04,
                f"{t0:.2f}-{t1:.2f} s",
                transform=ax.transAxes,
                color="white" if not invert_image else "black",
                fontsize=8,
                ha="left",
                va="bottom",
            )

        if show_scale_bar and scale_bar_length_um is not None and pixel_size_um:
            scale_bar_length_px = float(scale_bar_length_um) / float(pixel_size_um)
            margin = float(scale_bar_margin_px)
            if scale_bar_position == "lower_left":
                x0 = margin
                x1 = x0 + scale_bar_length_px
                y = height - margin
                text_y = y - 6
                va = "bottom"
            elif scale_bar_position == "upper_left":
                x0 = margin
                x1 = x0 + scale_bar_length_px
                y = margin
                text_y = y + 6
                va = "top"
            elif scale_bar_position == "upper_right":
                x1 = width - margin
                x0 = x1 - scale_bar_length_px
                y = margin
                text_y = y + 6
                va = "top"
            else:
                x1 = width - margin
                x0 = x1 - scale_bar_length_px
                y = height - margin
                text_y = y - 6
                va = "bottom"
            ax.plot(
                [x0, x1],
                [y, y],
                color=scale_bar_color,
                lw=scale_bar_line_width,
                solid_capstyle="butt",
            )
            ax.text(
                (x0 + x1) / 2,
                text_y,
                scale_bar_label or f"{scale_bar_length_um:g} um",
                color=scale_bar_color,
                fontsize=scale_bar_fontsize,
                ha="center",
                va=va,
            )

        out_path = os.path.join(
            output_dir,
            f"{output_prefix}_{raw_mode}_{i:03d}_frames_{window_start:05d}_{window_end:05d}.{output_format}",
        )
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0, format=output_format)
        plt.close(fig)
        output_paths.append(out_path)

    return {
        "output_paths": output_paths,
        "raw_movie_path": raw_movie_path,
        "frame_range": (frame_start, frame_end),
        "n_stills": len(output_paths),
        "raw_mode": raw_mode,
        "track_history": track_history,
        "output_format": output_format,
        "contour": contour,
        "raw_frame_offset": raw_frame_offset,
        "spatial_transform": spatial_transform,
    }


def _transform_xy_for_raw_alignment(x, y, width, height, transform="identity"):
    """Transform x/y display coordinates for raw-image alignment diagnostics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if transform in {None, "identity"}:
        return x, y
    if transform == "flip_x":
        return width - 1 - x, y
    if transform == "flip_y":
        return x, height - 1 - y
    if transform == "flip_xy":
        return width - 1 - x, height - 1 - y
    if transform == "transpose":
        return y, x
    if transform == "transpose_flip_x":
        return height - 1 - y, x
    if transform == "transpose_flip_y":
        return y, width - 1 - x
    if transform == "transpose_flip_xy":
        return height - 1 - y, width - 1 - x
    raise ValueError(
        "transform must be one of identity, flip_x, flip_y, flip_xy, "
        "transpose, transpose_flip_x, transpose_flip_y, transpose_flip_xy"
    )


def _crop_image_around_center(image, center_x, center_y, crop_width, crop_height, fill_value=0):
    """Crop image around center, padding outside the image with fill_value."""
    image = np.asarray(image)
    crop_width = int(max(1, round(crop_width)))
    crop_height = int(max(1, round(crop_height)))
    x0 = int(round(float(center_x) - crop_width / 2))
    y0 = int(round(float(center_y) - crop_height / 2))
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    out = np.full((crop_height, crop_width), fill_value, dtype=image.dtype)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.shape[1], x1)
    src_y1 = min(image.shape[0], y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        out[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = image[src_y0:src_y1, src_x0:src_x1]
    return out, x0, y0


def _select_track_ids_for_follow_view(
    tracks,
    *,
    track_id_col="unique_id",
    frame_col="frame",
    selection_mode="specified",
    track_ids=None,
    population_col=None,
    populations=None,
    n_tracks=1,
    random_state=42,
):
    """Select track IDs for follow/center single-molecule rendering."""
    if selection_mode == "specified":
        if track_ids is None:
            raise ValueError("track_ids is required when selection_mode='specified'")
        return list(track_ids)

    summary_source = tracks
    if selection_mode == "population":
        if population_col is None:
            raise ValueError("population_col is required when selection_mode='population'")
        if population_col not in tracks.columns:
            raise ValueError(f"Column {population_col!r} not found")
        if populations is None:
            raise ValueError("populations is required when selection_mode='population'")
        populations = [populations] if isinstance(populations, str) else list(populations)
        summary_source = tracks[tracks[population_col].isin(populations)].copy()
    elif selection_mode not in {"random", "all"}:
        raise ValueError("selection_mode must be 'specified', 'random', 'population', or 'all'")

    if summary_source.empty:
        raise ValueError("No tracks available after follow-view selection filters")

    summary = (
        summary_source.groupby(track_id_col)[frame_col]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    summary["track_length_frames"] = summary["max"] - summary["min"] + 1
    summary = summary.sort_values(["track_length_frames", "count"], ascending=False)

    if selection_mode == "all":
        return summary[track_id_col].tolist()
    if selection_mode == "random":
        return (
            summary.sample(
                n=min(int(n_tracks), len(summary)), random_state=random_state
            )[track_id_col]
            .tolist()
        )
    return summary.head(int(n_tracks))[track_id_col].tolist()


def render_follow_track_views(
    single_cell_df,
    tracks_df,
    *,
    output_dir,
    frame_range,
    master_dir=None,
    raw_movie_path=None,
    render_outputs=("png",),
    view_mode="follow",
    track_selection_mode="specified",
    track_ids=None,
    population_col=None,
    populations=None,
    n_tracks=1,
    random_state=42,
    track_id_col="unique_id",
    frame_col="frame",
    x_col="x",
    y_col="y",
    spatial_transform="identity",
    raw_frame_offset=0,
    crop_size_px=None,
    min_crop_size_px=64,
    crop_padding_px=12,
    frame_step=1,
    raw_mode="final_frame",
    output_format="png",
    dpi=200,
    fps=20,
    raw_cmap="gray",
    raw_lut=None,
    contrast_limits=None,
    invert_raw=False,
    track_history="cumulative",
    history_length_frames=None,
    track_color="orange",
    track_line_width=1.0,
    head_color="cyan",
    head_size=10,
    show_time_label=True,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    show_scale_bar=True,
    scale_bar_length_um=2.0,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    scale_bar_position="lower_right",
    scale_bar_margin_px=8,
    scale_bar_line_width=2,
    scale_bar_color="white",
    scale_bar_label=None,
    scale_bar_fontsize=7,
    napari_tail_length=60,
    close_after_save=True,
):
    """
    Render per-track views that either follow a molecule or use one fixed crop.

    ``view_mode='follow'`` keeps the selected molecule centered in each output
    frame. ``view_mode='center'`` uses one fixed crop centered on the selected
    track centroid, padded around the track extent.

    ``render_outputs`` can include ``"png"`` and/or ``"napari"``. For SVG stills,
    set ``output_format='svg'``; the raw image remains raster inside the SVG while
    overlays remain vector.
    """
    if view_mode not in {"follow", "center"}:
        raise ValueError("view_mode must be 'follow' or 'center'")
    render_outputs = [render_outputs] if isinstance(render_outputs, str) else list(render_outputs)
    for output in render_outputs:
        if output not in {"png", "napari"}:
            raise ValueError("render_outputs entries must be 'png' or 'napari'")
    output_format = str(output_format).lower().lstrip(".")
    if output_format not in {"png", "svg"}:
        raise ValueError("output_format must be 'png' or 'svg'")
    if raw_mode not in {"final_frame", "sum_window"}:
        raise ValueError("raw_mode must be 'final_frame' or 'sum_window'")

    cell_pdf = _as_pandas_copy(single_cell_df)
    tracks = _as_pandas_copy(tracks_df)
    for col in [track_id_col, frame_col, x_col, y_col]:
        if col not in tracks.columns:
            raise ValueError(f"Column {col!r} not found in tracks_df")

    frame_start, frame_end = map(int, frame_range)
    tracks = tracks[
        (tracks[frame_col] >= frame_start) & (tracks[frame_col] <= frame_end)
    ].copy()
    if tracks.empty:
        raise ValueError("No tracks available in frame_range")

    raw_movie_path = raw_movie_path or _resolve_raw_movie_path(cell_pdf, master_dir=master_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_track_color = track_color
    if isinstance(track_color, (list, tuple)):
        resolved_track_color = track_color[0] if len(track_color) else "orange"
    elif isinstance(track_color, str) and not is_color_like(track_color):
        try:
            resolved_track_color = plt.get_cmap(track_color)(0.15)
        except Exception:
            resolved_track_color = "orange"

    selected_ids = _select_track_ids_for_follow_view(
        tracks,
        track_id_col=track_id_col,
        frame_col=frame_col,
        selection_mode=track_selection_mode,
        track_ids=track_ids,
        population_col=population_col,
        populations=populations,
        n_tracks=n_tracks,
        random_state=random_state,
    )

    output_records = []
    cmap = _coerce_lut_colormap(raw_lut, target="matplotlib", fallback=raw_cmap)
    frame_step = max(1, int(frame_step))

    for track_id in selected_ids:
        track = tracks[tracks[track_id_col] == track_id].sort_values(frame_col).copy()
        if track.empty:
            continue

        track_frames = track[frame_col].astype(int).to_numpy()
        display_frames = np.arange(
            max(frame_start, int(track_frames.min())),
            min(frame_end, int(track_frames.max())) + 1,
            frame_step,
            dtype=int,
        )
        if display_frames.size == 0:
            continue

        raw_frames_needed = np.unique(display_frames + int(raw_frame_offset))
        if np.any(raw_frames_needed < 0):
            raise ValueError("raw_frame_offset makes a requested raw frame negative")
        raw_stack = _load_raw_frame_indices(raw_movie_path, raw_frames_needed)
        if raw_stack.ndim == 2:
            raw_stack = raw_stack[np.newaxis, ...]
        raw_by_frame = {int(frame): raw_stack[i] for i, frame in enumerate(raw_frames_needed)}
        height, width = raw_stack.shape[-2:]

        tx, ty = _transform_xy_for_raw_alignment(
            track[x_col].to_numpy(),
            track[y_col].to_numpy(),
            width,
            height,
            spatial_transform,
        )
        track = track.assign(_display_x=tx, _display_y=ty)
        interp_x = np.interp(display_frames, track_frames, tx)
        interp_y = np.interp(display_frames, track_frames, ty)

        if crop_size_px is None:
            track_width = float(np.nanmax(tx) - np.nanmin(tx))
            track_height = float(np.nanmax(ty) - np.nanmin(ty))
            auto_width = max(min_crop_size_px, int(np.ceil(track_width + 2 * crop_padding_px)))
            auto_height = max(min_crop_size_px, int(np.ceil(track_height + 2 * crop_padding_px)))
        elif isinstance(crop_size_px, (list, tuple)):
            auto_width, auto_height = map(int, crop_size_px)
        else:
            auto_width = auto_height = int(crop_size_px)

        if view_mode == "center":
            fixed_center_x = float(np.nanmean(tx))
            fixed_center_y = float(np.nanmean(ty))
            centers_x = np.full_like(display_frames, fixed_center_x, dtype=float)
            centers_y = np.full_like(display_frames, fixed_center_y, dtype=float)
        else:
            centers_x = interp_x
            centers_y = interp_y

        track_label = str(track_id).replace(os.sep, "_").replace(" ", "_")
        track_dir = output_dir / f"track_{track_label}_{view_mode}"
        png_dir = track_dir / f"{output_format}_frames"
        if "png" in render_outputs:
            png_dir.mkdir(parents=True, exist_ok=True)

        cropped_stack = []
        napari_rows = []
        origins = {}
        for idx, frame in enumerate(display_frames):
            raw_frame = int(frame + int(raw_frame_offset))
            raw = np.asarray(raw_by_frame[raw_frame])
            if raw_mode == "sum_window":
                # Sum from previous displayed frame to this frame for streak-style raw view.
                prior_frame = display_frames[max(0, idx - 1)]
                sum_frames = np.arange(prior_frame, frame + 1) + int(raw_frame_offset)
                sum_stack = _load_raw_frame_indices(raw_movie_path, sum_frames)
                raw = np.nansum(sum_stack, axis=0) if sum_stack.ndim == 3 else sum_stack
            if invert_raw:
                raw = np.nanmax(raw) - raw
            raw_norm = _normalize_raw_image(
                raw, contrast_limits=contrast_limits, invert_image=False
            )
            crop, origin_x, origin_y = _crop_image_around_center(
                raw_norm,
                centers_x[idx],
                centers_y[idx],
                auto_width,
                auto_height,
                fill_value=float(np.nanmin(raw_norm)),
            )
            cropped_stack.append(crop)
            origins[int(frame)] = (origin_x, origin_y)

            if track_history == "window" and history_length_frames is not None:
                history_start = max(frame_start, int(frame - history_length_frames))
            elif track_history == "window":
                history_start = max(frame_start, int(display_frames[max(0, idx - 1)]))
            elif history_length_frames is not None:
                history_start = max(frame_start, int(frame - history_length_frames))
            else:
                history_start = frame_start

            visible = track[
                (track[frame_col] >= history_start) & (track[frame_col] <= frame)
            ].copy()
            if "png" in render_outputs:
                fig, ax = plt.subplots(figsize=(3, 3), dpi=dpi)
                ax.imshow(crop, cmap=cmap, origin="upper")
                if len(visible) >= 2:
                    ax.plot(
                        visible["_display_x"] - origin_x,
                        visible["_display_y"] - origin_y,
                        color=resolved_track_color,
                        lw=track_line_width,
                        alpha=0.95,
                    )
                current = visible.tail(1)
                if not current.empty:
                    ax.scatter(
                        current["_display_x"] - origin_x,
                        current["_display_y"] - origin_y,
                        color=head_color,
                        s=head_size,
                        edgecolors="none",
                        zorder=3,
                    )
                ax.set_xlim(0, auto_width)
                ax.set_ylim(auto_height, 0)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                if show_time_label:
                    ax.text(
                        0.02,
                        0.04,
                        f"{frame * frame_interval:.2f} s",
                        transform=ax.transAxes,
                        color=scale_bar_color,
                        fontsize=scale_bar_fontsize,
                        ha="left",
                        va="bottom",
                    )
                if show_scale_bar and scale_bar_length_um is not None and pixel_size_um:
                    bar_len_px = float(scale_bar_length_um) / float(pixel_size_um)
                    margin = float(scale_bar_margin_px)
                    if scale_bar_position == "lower_left":
                        x0 = margin
                        x1 = x0 + bar_len_px
                        y = auto_height - margin
                        text_y = y - 5
                        va = "bottom"
                    elif scale_bar_position == "upper_left":
                        x0 = margin
                        x1 = x0 + bar_len_px
                        y = margin
                        text_y = y + 5
                        va = "top"
                    elif scale_bar_position == "upper_right":
                        x1 = auto_width - margin
                        x0 = x1 - bar_len_px
                        y = margin
                        text_y = y + 5
                        va = "top"
                    else:
                        x1 = auto_width - margin
                        x0 = x1 - bar_len_px
                        y = auto_height - margin
                        text_y = y - 5
                        va = "bottom"
                    ax.plot([x0, x1], [y, y], color=scale_bar_color, lw=scale_bar_line_width)
                    ax.text(
                        (x0 + x1) / 2,
                        text_y,
                        scale_bar_label or f"{scale_bar_length_um:g} um",
                        color=scale_bar_color,
                        fontsize=scale_bar_fontsize,
                        ha="center",
                        va=va,
                    )
                out_path = png_dir / f"track_{track_label}_{view_mode}_{idx:04d}_frame_{frame:05d}.{output_format}"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0, format=output_format)
                plt.close(fig)

            # Napari stores each point in the crop coordinates of its own frame.
            frame_points = track[track[frame_col] == frame]
            for _, row in frame_points.iterrows():
                napari_rows.append(
                    [
                        0,
                        int(np.where(display_frames == frame)[0][0]),
                        float(row["_display_y"] - origin_y),
                        float(row["_display_x"] - origin_x),
                    ]
                )

        movie_path = None
        viewer = None
        if "napari" in render_outputs:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(
                np.asarray(cropped_stack),
                name=f"Raw follow view {track_id}",
                colormap=_coerce_lut_colormap(raw_lut, target="napari", fallback=raw_cmap),
                contrast_limits=None,
            )
            if napari_rows:
                napari_tracks = pd.DataFrame(napari_rows, columns=["track_id", "frame", "y", "x"])
                viewer.add_tracks(
                    napari_tracks.to_numpy(),
                    name=f"Track {track_id}",
                    tail_length=napari_tail_length,
                    tail_width=max(1, int(round(track_line_width))),
                )
                movie_path = track_dir / f"track_{track_label}_{view_mode}.mp4"
                save_movie(
                    viewer,
                    napari_tracks,
                    save_path=str(movie_path),
                    steps=max(1, len(display_frames) - 1),
                    fps=fps,
                )
            if close_after_save:
                viewer.close()

        output_records.append(
            {
                "track_id": track_id,
                "view_mode": view_mode,
                "n_frames": int(len(display_frames)),
                "frame_start": int(display_frames.min()),
                "frame_end": int(display_frames.max()),
                "crop_width": int(auto_width),
                "crop_height": int(auto_height),
                "png_dir": str(png_dir) if "png" in render_outputs else None,
                "movie_path": str(movie_path) if movie_path is not None else None,
            }
        )

    return {
        "selected_track_ids": selected_ids,
        "outputs": output_records,
        "raw_movie_path": raw_movie_path,
        "frame_range": (frame_start, frame_end),
    }
