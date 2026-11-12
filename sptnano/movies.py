"""State-change track movies (track coloured by behaviour state, flashes at switches)."""

import imageio
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config
from . import superwindows
from .raw_images import (
    load_fiji_lut_collection,
    _coerce_lut_colormap,
    _as_pandas_copy,
    _single_value_from_df,
    _resolve_raw_movie_path,
    _normalize_raw_image,
    _load_raw_frame_indices,
)


GOLD_RGB = (1.0, 0.84, 0.15)


STATE_MOVIE_FLASH_MODES = ("lut_whiten", "head_pulse", "track_gold")


STATE_MOVIE_VIEW_MODES = ("superwindow", "whole_track")


STATE_MOVIE_FOV_MODES = ("track_crop", "particle", "full")


STATE_COL_CANDIDATES = (
    "superwindow_window_state_for_visualization",
    "transformer_behaviour_state",
    "transformer_behavior_state",
    "superwindow_resolved_state",
    "amalgam_behaviour_state",
    "final_population",
)


TRACK_ID_CANDIDATES = ("unique_id", "original_track_id", "track_id", "particle")


def _first_present_column(columns, candidates, kind="column"):
    cols = set(columns)
    for name in candidates:
        if name in cols:
            return name
    raise KeyError(f"No {kind} column found. Tried {list(candidates)}")


def _as_rgba_tuple(color, alpha=1.0):
    rgba = mpl.colors.to_rgba(color, alpha=alpha)
    return tuple(float(v) for v in rgba)


def _pulse_envelope(n_frames, kind="triangle"):
    n = max(1, int(n_frames))
    if n == 1:
        return np.array([1.0], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    if kind == "cosine":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * x)
    return 1.0 - np.abs(2.0 * x - 1.0)


def _whiten_cmap(cmap, amount, name="lut_flash"):
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return cmap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    colors = np.asarray(cmap(np.linspace(0.0, 1.0, 256)), dtype=float)
    colors[:, :3] = colors[:, :3] * (1.0 - amount) + amount
    return LinearSegmentedColormap.from_list(name, colors)


def _sequence_state_stats(states):
    states = [str(s) for s in list(states) if pd.notna(s)]
    n = len(states)
    if n == 0:
        return {
            "n_steps": 0,
            "n_distinct_states": 0,
            "n_switches": 0,
            "state_set": "",
            "entropy": 0.0,
            "mix_score": 0.0,
            "dominant_state": None,
            "dominant_fraction": 0.0,
        }
    seen = []
    for state in states:
        if state not in seen:
            seen.append(state)
    n_switches = int(sum(states[i] != states[i - 1] for i in range(1, n)))
    counts = pd.Series(states).value_counts(normalize=True)
    probs = counts.to_numpy(dtype=float)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    stats = {
        "n_steps": n,
        "n_distinct_states": len(seen),
        "n_switches": n_switches,
        "state_set": ",".join(seen),
        "entropy": entropy,
        "mix_score": float(n_switches) * (1.0 + entropy),
        "dominant_state": str(counts.idxmax()),
        "dominant_fraction": float(counts.max()),
    }
    for state, frac in counts.items():
        stats[f"frac_{state}"] = float(frac)
    return stats


def _copy_unit_metadata(row, columns):
    return {col: row[col] for col in columns if col in row.index}


def summarize_state_units_for_movie(
    superwindow_df=None,
    instant_df=None,
    *,
    view_mode="superwindow",
    state_col=None,
    track_id_col=None,
    frame_col="frame",
    state_sequence_col="state_sequence",
    metadata_cols=None,
):
    """
    Per-superwindow or per-track state composition table for movie filtering.

    ``view_mode='superwindow'`` scores each ``state_sequence``.
    ``view_mode='whole_track'`` scores each track after de-duplicating frames.
    """
    view_mode = str(view_mode).strip().lower()
    if view_mode not in STATE_MOVIE_VIEW_MODES:
        raise ValueError(f"view_mode must be one of {STATE_MOVIE_VIEW_MODES}")

    default_meta = [
        "filename",
        "condition",
        "foldername",
        "geno",
        "mol",
        "group",
        "location",
        "type",
        "superwindow_length",
        "original_track_id",
        "unique_id",
    ]
    metadata_cols = list(metadata_cols or default_meta)
    rows = []

    if view_mode == "superwindow":
        if superwindow_df is None:
            raise ValueError("superwindow_df is required for view_mode='superwindow'")
        sw = _as_pandas_copy(superwindow_df)
        if state_sequence_col not in sw.columns:
            raise KeyError(f"{state_sequence_col!r} not found in superwindow_df")
        if "superwindow_id" not in sw.columns:
            raise KeyError("superwindow_df must contain 'superwindow_id'")
        for _, row in sw.iterrows():
            stats = _sequence_state_stats(row[state_sequence_col])
            item = {
                "unit_id": row["superwindow_id"],
                "view_mode": view_mode,
                "track_id": row.get("original_track_id", row.get("unique_id")),
            }
            item.update(_copy_unit_metadata(row, metadata_cols))
            item.update(stats)
            rows.append(item)
        return pd.DataFrame(rows)

    if instant_df is None:
        raise ValueError("instant_df is required for view_mode='whole_track'")
    instant = _as_pandas_copy(instant_df)
    track_id_col = track_id_col or _first_present_column(instant.columns, TRACK_ID_CANDIDATES, "track id")
    state_col = state_col or _first_present_column(instant.columns, STATE_COL_CANDIDATES, "state")
    if frame_col not in instant.columns:
        raise KeyError(f"{frame_col!r} not found in instant_df")
    for track_id, sub in instant.groupby(track_id_col, sort=False):
        sub = sub.sort_values(frame_col).drop_duplicates(frame_col, keep="last")
        stats = _sequence_state_stats(sub[state_col])
        first = sub.iloc[0]
        item = {
            "unit_id": track_id,
            "view_mode": view_mode,
            "track_id": track_id,
            "n_frames": int(len(sub)),
        }
        item.update(_copy_unit_metadata(first, metadata_cols))
        item.update(stats)
        rows.append(item)
    return pd.DataFrame(rows)


def filter_state_units_for_movie(
    summary_df,
    *,
    target_state=None,
    min_target_fraction=0.0,
    max_target_fraction=1.0,
    require_states=None,
    exclude_states=None,
    min_n_distinct_states=1,
    max_n_distinct_states=None,
    min_n_switches=0,
    max_n_switches=None,
    min_steps=None,
    group_col=None,
    group_values=None,
    length=None,
    length_col="superwindow_length",
    prefer="most_switches",
    n_select=1,
    random_state=0,
    exclude_ids=None,
):
    """
    Filter and rank the table from ``summarize_state_units_for_movie``.

    ``prefer`` is one of: ``most_switches``, ``most_mixed``, ``most_of_state``,
    ``longest``, ``fewest_switches``, ``random``.
    """
    df = _as_pandas_copy(summary_df)
    if df.empty:
        raise ValueError("summary_df is empty")
    if exclude_ids:
        skip = {str(x) for x in list(exclude_ids)}
        df = df.loc[~df["unit_id"].astype(str).isin(skip)].copy()
        if df.empty:
            raise ValueError("No units left after exclude_ids")

    if group_col is not None and group_values is not None:
        if group_col not in df.columns:
            raise KeyError(f"{group_col!r} not found in summary table")
        keep = {str(v) for v in list(group_values)}
        df = df.loc[df[group_col].astype(str).isin(keep)].copy()

    if length is not None and length_col in df.columns:
        df = df.loc[df[length_col].astype(float) == float(length)].copy()

    if min_steps is not None:
        step_col = "n_frames" if "n_frames" in df.columns and df["n_frames"].notna().any() else "n_steps"
        df = df.loc[df[step_col] >= int(min_steps)].copy()

    require_states = [] if require_states is None else [str(s) for s in list(require_states)]
    exclude_states = [] if exclude_states is None else [str(s) for s in list(exclude_states)]

    def _state_set(value):
        if pd.isna(value) or value is None:
            return set()
        return {part for part in str(value).split(",") if part}

    if require_states:
        df = df.loc[df["state_set"].map(lambda s: set(require_states).issubset(_state_set(s)))].copy()
    if exclude_states:
        df = df.loc[df["state_set"].map(lambda s: _state_set(s).isdisjoint(exclude_states))].copy()

    df = df.loc[
        (df["n_distinct_states"] >= int(min_n_distinct_states))
        & (df["n_switches"] >= int(min_n_switches))
    ].copy()
    if max_n_distinct_states is not None:
        df = df.loc[df["n_distinct_states"] <= int(max_n_distinct_states)].copy()
    if max_n_switches is not None:
        df = df.loc[df["n_switches"] <= int(max_n_switches)].copy()

    if target_state is not None:
        frac_col = f"frac_{target_state}"
        if frac_col not in df.columns:
            df[frac_col] = 0.0
        df[frac_col] = df[frac_col].fillna(0.0)
        df = df.loc[
            (df[frac_col] >= float(min_target_fraction))
            & (df[frac_col] <= float(max_target_fraction))
        ].copy()

    if df.empty:
        raise ValueError("No units survived the state-composition filters")

    prefer = str(prefer).strip().lower()
    rng = np.random.default_rng(int(random_state))
    if prefer == "most_switches":
        df = df.sort_values(["n_switches", "mix_score", "n_steps"], ascending=False)
    elif prefer == "most_mixed":
        df = df.sort_values(["mix_score", "n_distinct_states", "n_switches"], ascending=False)
    elif prefer == "most_of_state":
        if target_state is None:
            raise ValueError("prefer='most_of_state' requires target_state")
        df = df.sort_values([f"frac_{target_state}", "n_steps"], ascending=False)
    elif prefer == "longest":
        length_sort = "n_frames" if "n_frames" in df.columns else "n_steps"
        df = df.sort_values([length_sort, "n_switches"], ascending=False)
    elif prefer == "fewest_switches":
        df = df.sort_values(["n_switches", "n_distinct_states", "n_steps"], ascending=[True, True, False])
    elif prefer == "random":
        df = df.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    else:
        raise ValueError(
            "prefer must be one of: most_switches, most_mixed, most_of_state, "
            "longest, fewest_switches, random"
        )

    selected = df.head(int(n_select)).copy() if prefer != "random" else df.iloc[
        rng.choice(len(df), size=min(int(n_select), len(df)), replace=False)
    ].copy()
    return {
        "summary_df": df.reset_index(drop=True),
        "selected_df": selected.reset_index(drop=True),
        "selected_ids": selected["unit_id"].tolist(),
    }


def _map_superwindow_instant(superwindow_df, instant_df, unit_id, pixel_size_um=None, visual_state_col=None):
    visual_state_col = visual_state_col or "superwindow_window_state_for_visualization"
    window_state_map = superwindows.build_window_state_map_from_superwindows(
        superwindow_df,
        [unit_id],
        visual_state_col=visual_state_col,
    )
    mapped = superwindows.map_window_states_to_instant(
        instant_df,
        window_state_map,
        visual_state_col=visual_state_col,
        pixel_size_um=pixel_size_um,
    )
    return mapped, visual_state_col


def _build_frame_track(
    instant_df,
    *,
    view_mode,
    unit_id,
    superwindow_df=None,
    state_col=None,
    track_id_col=None,
    frame_col="frame",
    x_col=None,
    y_col=None,
    pixel_size_um=None,
    overlap_policy="later_window",
):
    """Return one row per movie frame with pixel xy and a state label."""
    if view_mode == "superwindow":
        mapped, visual_state_col = _map_superwindow_instant(
            superwindow_df,
            instant_df,
            unit_id,
            pixel_size_um=pixel_size_um,
            visual_state_col=state_col,
        )
        state_col = visual_state_col
        track = mapped.copy()
        sort_cols = [c for c in ["position_in_superwindow", frame_col] if c in track.columns]
        track = track.sort_values(sort_cols)
        keep = "last" if overlap_policy == "later_window" else "first"
        track = track.drop_duplicates(frame_col, keep=keep).sort_values(frame_col)
    else:
        instant = _as_pandas_copy(instant_df)
        track_id_col = track_id_col or _first_present_column(instant.columns, TRACK_ID_CANDIDATES, "track id")
        state_col = state_col or _first_present_column(instant.columns, STATE_COL_CANDIDATES, "state")
        track = instant.loc[instant[track_id_col].astype(str) == str(unit_id)].copy()
        if track.empty:
            raise ValueError(f"unit_id {unit_id!r} not found in instant_df[{track_id_col}]")
        track = track.sort_values(frame_col).drop_duplicates(frame_col, keep="last")

    if x_col is None:
        x_col = "x" if "x" in track.columns else "x_um"
    if y_col is None:
        y_col = "y" if "y" in track.columns else "y_um"
    for col in [x_col, y_col, frame_col, state_col]:
        if col not in track.columns:
            raise KeyError(f"{col!r} not found after building the frame track")

    track = track.sort_values(frame_col).reset_index(drop=True)
    x = track[x_col].to_numpy(dtype=float)
    y = track[y_col].to_numpy(dtype=float)
    if x_col.endswith("_um") or y_col.endswith("_um"):
        px = float(pixel_size_um if pixel_size_um is not None else getattr(config, "PIXELSIZE_MICRONS", 0.065))
        x = x / px
        y = y / px
    out = pd.DataFrame(
        {
            "frame": track[frame_col].astype(int).to_numpy(),
            "x": x,
            "y": y,
            "state": track[state_col].astype(str).to_numpy(),
        }
    )
    if "time_s" in track.columns:
        out["time_s"] = track["time_s"].to_numpy(dtype=float)
    change = np.zeros(len(out), dtype=bool)
    if len(out) > 1:
        change[1:] = out["state"].to_numpy()[1:] != out["state"].to_numpy()[:-1]
    out["is_state_change"] = change
    out["from_state"] = out["state"].shift(1)
    out.loc[out["is_state_change"], "change_label"] = (
        out.loc[out["is_state_change"], "from_state"].astype(str)
        + " → "
        + out.loc[out["is_state_change"], "state"].astype(str)
    )
    meta_cols = [
        c
        for c in ("filename", "condition", "foldername", "geno", "unique_id", "original_track_id", "superwindow_id")
        if c in track.columns
    ]
    return out, track.iloc[0][meta_cols] if meta_cols else pd.Series(dtype=object), state_col


def _expand_flash_timeline(track_df, flash_hold_frames=6, envelope="triangle"):
    weights = _pulse_envelope(flash_hold_frames, kind=envelope)
    rows = []
    output_i = 0
    for source_index, row in track_df.iterrows():
        n_hold = int(flash_hold_frames) if bool(row["is_state_change"]) else 1
        local_weights = weights if n_hold > 1 else np.array([0.0])
        for hold_i in range(n_hold):
            item = row.to_dict()
            item["source_index"] = int(source_index)
            item["output_index"] = output_i
            item["flash_weight"] = float(local_weights[hold_i] if n_hold > 1 else 0.0)
            item["is_flash_hold"] = n_hold > 1
            rows.append(item)
            output_i += 1
    return pd.DataFrame(rows)


def _smooth_track_coordinates(track_df, smooth=0.0):
    """Gaussian-smooth x/y in place (same rows, states unchanged). ``smooth`` is sigma in points."""
    out = track_df.copy()
    sigma = float(smooth)
    if sigma <= 0 or len(out) < 3:
        return out
    from scipy.ndimage import gaussian_filter1d

    out["x"] = gaussian_filter1d(out["x"].to_numpy(dtype=float), sigma=sigma, mode="nearest")
    out["y"] = gaussian_filter1d(out["y"].to_numpy(dtype=float), sigma=sigma, mode="nearest")
    return out


def _apply_view_zoom(x0, x1, y0, y1, zoom, image_shape=None):
    """``zoom>1`` zooms in (smaller FOV). ``zoom<1`` zooms out."""
    zoom = max(float(zoom), 0.05)
    xc = 0.5 * (float(x0) + float(x1))
    yc = 0.5 * (float(y0) + float(y1))
    half_w = 0.5 * (float(x1) - float(x0)) / zoom
    half_h = 0.5 * (float(y1) - float(y0)) / zoom
    half = max(half_w, half_h, 4.0)
    x0, x1 = xc - half, xc + half
    y0, y1 = yc - half, yc + half
    if image_shape is not None:
        height, width = image_shape[-2:]
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(width), x1)
        y1 = min(float(height), y1)
    return int(np.floor(x0)), int(np.ceil(x1)), int(np.floor(y0)), int(np.ceil(y1))


def _track_crop_bounds(x, y, *, pad_px=12, image_shape=None, square=True):
    x0 = float(np.nanmin(x)) - pad_px
    x1 = float(np.nanmax(x)) + pad_px
    y0 = float(np.nanmin(y)) - pad_px
    y1 = float(np.nanmax(y)) + pad_px
    if square:
        width = max(x1 - x0, y1 - y0, 8.0)
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        x0, x1 = xc - width / 2.0, xc + width / 2.0
        y0, y1 = yc - width / 2.0, yc + width / 2.0
    if image_shape is not None:
        height, width = image_shape[-2:]
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(width), x1)
        y1 = min(float(height), y1)
    return int(np.floor(x0)), int(np.ceil(x1)), int(np.floor(y0)), int(np.ceil(y1))


def _particle_crop_bounds(x, y, box_size, image_shape=None):
    half = int(box_size) // 2
    xc = int(round(float(x)))
    yc = int(round(float(y)))
    x0, x1 = xc - half, xc - half + int(box_size)
    y0, y1 = yc - half, yc - half + int(box_size)
    if image_shape is not None:
        height, width = image_shape[-2:]
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > width:
            x0 = max(0, x0 - (x1 - width))
            x1 = width
        if y1 > height:
            y0 = max(0, y0 - (y1 - height))
            y1 = height
    return x0, x1, y0, y1


def _safe_raw_movie_path(meta_row, master_dir=None, raw_movie_path=None):
    if raw_movie_path:
        return str(raw_movie_path)
    meta_df = meta_row.to_frame().T if isinstance(meta_row, pd.Series) else _as_pandas_copy(meta_row)
    candidates = []
    foldername = _single_value_from_df(meta_df, "foldername")
    for candidate_master in (foldername, master_dir):
        if not candidate_master:
            continue
        try:
            path = _resolve_raw_movie_path(meta_df, master_dir=candidate_master)
        except Exception:
            continue
        candidates.append(path)
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else None


def _load_raw_frames_for_movie(raw_movie_path, frames):
    if raw_movie_path is None or not os.path.exists(str(raw_movie_path)):
        return None, None
    unique_frames = np.unique(np.asarray(frames, dtype=int))
    stack = _load_raw_frame_indices(raw_movie_path, unique_frames)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    frame_to_index = {int(f): i for i, f in enumerate(unique_frames)}
    return stack, frame_to_index


def _add_scale_bar(ax, length_um, pixel_size_um, color="white", linewidth=2.2, fontsize=8):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    length_px = float(length_um) / float(pixel_size_um)
    x0 = min(xlim) + 0.08 * abs(xlim[1] - xlim[0])
    y_span = abs(ylim[1] - ylim[0])
    # Image-style inverted y: larger y is visually lower.
    if ylim[0] > ylim[1]:
        y0 = ylim[0] - 0.10 * y_span
    else:
        y0 = ylim[0] + 0.10 * y_span
    ax.plot(
        [x0, x0 + length_px],
        [y0, y0],
        color=color,
        lw=linewidth,
        solid_capstyle="butt",
        zorder=8,
        clip_on=True,
    )
    ax.text(
        x0 + 0.5 * length_px,
        y0,
        f"{length_um:g} µm",
        color=color,
        fontsize=fontsize,
        ha="center",
        va="bottom",
        zorder=8,
        clip_on=True,
    )


def _hide_axes_chrome(ax):
    """Seaborn whitegrid otherwise leaves grid/spine strokes on the track."""
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_edgecolor("none")
    ax.patch.set_linewidth(0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _curved_polyline(x, y, states, samples_per_seg=1):
    """Return denser coordinates, keeping the state of the originating segment."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    states = np.asarray(states)
    n = len(x)
    samples_per_seg = max(1, int(samples_per_seg))
    if n < 2 or samples_per_seg == 1:
        return x, y, states
    if n < 4:
        return x, y, states
    from scipy.interpolate import interp1d

    t = np.arange(n, dtype=float)
    t_new = np.linspace(0.0, n - 1.0, (n - 1) * samples_per_seg + 1)
    kind = "cubic"
    xs = interp1d(t, x, kind=kind, bounds_error=False, fill_value="extrapolate")(t_new)
    ys = interp1d(t, y, kind=kind, bounds_error=False, fill_value="extrapolate")(t_new)
    src = np.clip(np.floor(t_new).astype(int), 0, n - 2)
    return xs, ys, states[src]


def _draw_state_colored_trail(
    ax,
    vis,
    *,
    x0,
    y0,
    state_colors,
    track_linewidth,
    gold,
    flash_weight=0.0,
    draw_gold=False,
    curve_samples=1,
):
    """Draw the full visible trail as connected segments (no frame-gap skips)."""
    if vis is None or len(vis) < 2:
        return
    vx = vis["x"].to_numpy(dtype=float) - float(x0)
    vy = vis["y"].to_numpy(dtype=float) - float(y0)
    states = vis["state"].astype(str).to_numpy()
    vx, vy, states = _curved_polyline(vx, vy, states, samples_per_seg=curve_samples)
    gold_lw = track_linewidth * (2.4 + 2.6 * flash_weight)
    gold_alpha = 0.18 + 0.45 * flash_weight
    for i in range(len(vx) - 1):
        xs = (vx[i], vx[i + 1])
        ys = (vy[i], vy[i + 1])
        if draw_gold and flash_weight > 0:
            ax.plot(
                xs,
                ys,
                color=(*gold, gold_alpha),
                lw=gold_lw,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )
        ax.plot(
            xs,
            ys,
            color=state_colors.get(states[i], "0.7"),
            lw=track_linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )


def render_state_change_track_movie(
    instant_df,
    unit_id,
    *,
    view_mode="superwindow",
    superwindow_df=None,
    state_col=None,
    track_id_col=None,
    state_colors=None,
    state_order=None,
    overlay_on_raw=True,
    raw_movie_path=None,
    master_dir=None,
    use_fiji_lut=False,
    fiji_lut=None,
    fiji_lut_name=None,
    fiji_lut_dir=r"D:\GITHUB_SOFTWARE\Fiji.app\luts",
    fiji_lut_prefix="JDM",
    raw_cmap="gray",
    invert_raw=False,
    background="black",
    raw_contrast_limits=None,
    fov_mode="track_crop",
    crop_pad_px=16,
    view_zoom=1.0,
    box_size=48,
    overlap_policy="later_window",
    flash_modes=("head_pulse", "track_gold"),
    flash_hold_frames=6,
    flash_envelope="triangle",
    lut_whiten_amount=0.55,
    head_base_radius_px=2.2,
    head_pulse_scale=2.4,
    gold_color=GOLD_RGB,
    track_linewidth=2.0,
    track_smooth=0.0,
    track_curve=1,
    trail="cumulative",
    pixel_size_um=None,
    frame_interval=None,
    show_timer=True,
    show_state_label=True,
    show_scale_bar=True,
    scale_bar_length_um=1.0,
    figsize=None,
    dpi=160,
    output_dir=None,
    png_prefix=None,
    write_png=True,
    write_gif=True,
    fps=20,
    preview_frames=True,
    show_plot=False,
):
    """
    Write a PNG-stack movie of one superwindow or whole track, colored by state.

    A state change inserts ``flash_hold_frames`` extra PNGs so the event is visible.
    ``flash_modes`` can combine:

    - ``lut_whiten``: raw LUT / canvas briefly lifts toward white
    - ``head_pulse``: current spot grows then shrinks, gold translucent ring
    - ``track_gold``: whole visible trail gets a gold glow/outline

    ``background`` is ``"black"`` or ``"white"`` and is used only when
    ``overlay_on_raw`` is False (or the raw TIFF cannot be loaded).

    ``track_linewidth`` is the trail thickness in plot points.
    ``track_smooth`` is a Gaussian sigma in frames (0 = off).
    ``track_curve`` is extra cubic samples per segment (1 = straight chords).
    ``view_zoom`` > 1 zooms in; < 1 zooms out. Also use ``crop_pad_px`` /
    ``box_size`` for FOV.
    """
    view_mode = str(view_mode).strip().lower()
    fov_mode = str(fov_mode).strip().lower()
    if view_mode not in STATE_MOVIE_VIEW_MODES:
        raise ValueError(f"view_mode must be one of {STATE_MOVIE_VIEW_MODES}")
    if fov_mode not in STATE_MOVIE_FOV_MODES:
        raise ValueError(f"fov_mode must be one of {STATE_MOVIE_FOV_MODES}")
    if isinstance(flash_modes, str):
        flash_modes = (flash_modes,)
    flash_modes = tuple(str(m).strip().lower() for m in flash_modes)
    unknown = [m for m in flash_modes if m not in STATE_MOVIE_FLASH_MODES]
    if unknown:
        raise ValueError(f"Unknown flash_modes {unknown}. Use {STATE_MOVIE_FLASH_MODES}")
    if trail not in {"cumulative", "current_state"}:
        raise ValueError("trail must be 'cumulative' or 'current_state'")
    background = str(background).strip().lower()
    if background not in {"black", "white"}:
        raise ValueError("background must be 'black' or 'white'")

    pixel_size_um = float(pixel_size_um if pixel_size_um is not None else getattr(config, "PIXELSIZE_MICRONS", 0.065))
    frame_interval = float(frame_interval if frame_interval is not None else getattr(config, "TIME_BETWEEN_FRAMES", 0.01))

    track_df, meta_row, used_state_col = _build_frame_track(
        instant_df,
        view_mode=view_mode,
        unit_id=unit_id,
        superwindow_df=superwindow_df,
        state_col=state_col,
        track_id_col=track_id_col,
        pixel_size_um=pixel_size_um,
        overlap_policy=overlap_policy,
    )
    if track_df.empty:
        raise ValueError(f"No frames available for unit_id={unit_id!r}")

    track_df = _smooth_track_coordinates(track_df, smooth=track_smooth)
    track_df = track_df.reset_index(drop=True)

    timeline = _expand_flash_timeline(
        track_df, flash_hold_frames=flash_hold_frames, envelope=flash_envelope
    )

    observed_states = [s for s in (state_order or []) if s in set(track_df["state"])]
    observed_states += [s for s in pd.unique(track_df["state"]) if s not in observed_states]
    if state_colors is None:
        cmap = plt.get_cmap("Dark2")
        state_colors = {
            state: mpl.colors.to_hex(cmap(i / max(len(observed_states) - 1, 1)))
            for i, state in enumerate(observed_states)
        }

    resolved_raw_path = None
    raw_stack = None
    frame_to_index = None
    raw_lut_obj = None
    if overlay_on_raw:
        if use_fiji_lut and fiji_lut is None:
            try:
                lut_dict, chosen = load_fiji_lut_collection(
                    lut_dir=fiji_lut_dir,
                    prefix=fiji_lut_prefix,
                    default_lut_name=fiji_lut_name or "JDM_Ink Wash Yellow",
                )
                raw_lut_obj = lut_dict.get(fiji_lut_name, chosen) if fiji_lut_name else chosen
            except Exception as exc:
                print(f"[state-movie] Fiji LUT load failed ({exc}); falling back to {raw_cmap!r}")
        else:
            raw_lut_obj = fiji_lut
        resolved_raw_path = _safe_raw_movie_path(meta_row, master_dir=master_dir, raw_movie_path=raw_movie_path)
        raw_stack, frame_to_index = _load_raw_frames_for_movie(resolved_raw_path, track_df["frame"])
        if raw_stack is None:
            print(f"[state-movie] Raw movie not found ({resolved_raw_path}); drawing track only.")
            overlay_on_raw = False

    image_shape = None if raw_stack is None else raw_stack.shape
    xs = track_df["x"].to_numpy(dtype=float)
    ys = track_df["y"].to_numpy(dtype=float)
    if fov_mode == "full" and image_shape is not None:
        height, width = image_shape[-2:]
        crop = (0, int(width), 0, int(height))
    elif fov_mode == "particle":
        crop = None
    else:
        crop = _track_crop_bounds(xs, ys, pad_px=crop_pad_px, image_shape=image_shape, square=True)
        if float(view_zoom) != 1.0:
            crop = _apply_view_zoom(*crop, zoom=view_zoom, image_shape=image_shape)

    if output_dir is None:
        tag = f"{view_mode}_{unit_id}"
        output_dir = os.path.join(str(master_dir or getattr(config, "MASTER", ".")), "movies", "state_change", tag)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if png_prefix is None:
        png_prefix = f"{view_mode}_{unit_id}".replace(os.sep, "_")

    base_cmap = _coerce_lut_colormap(raw_lut_obj, target="matplotlib", fallback=raw_cmap)
    gold = _as_rgba_tuple(gold_color, alpha=1.0)[:3]
    if overlay_on_raw:
        text_color = "black" if invert_raw else "white"
    else:
        text_color = "black" if background == "white" else "white"
    canvas_rgb = (1.0, 1.0, 1.0) if background == "white" else (0.0, 0.0, 0.0)
    png_paths = []
    gif_frames = []

    for _, shot in timeline.iterrows():
        end_i = int(shot["source_index"]) if "source_index" in shot else int(shot.name)
        vis = track_df.iloc[: end_i + 1]
        if trail == "current_state":
            vis = vis.loc[vis["state"] == shot["state"]]
        weight = float(shot["flash_weight"])
        frame_i = int(shot["frame"])
        head_x = float(vis["x"].iloc[-1]) if len(vis) else float(shot["x"])
        head_y = float(vis["y"].iloc[-1]) if len(vis) else float(shot["y"])

        if fov_mode == "particle":
            particle_box = max(8.0, float(box_size) / max(float(view_zoom), 0.05))
            x0, x1, y0, y1 = _particle_crop_bounds(head_x, head_y, particle_box, image_shape=image_shape)
        else:
            x0, x1, y0, y1 = crop

        crop_w = max(8, x1 - x0)
        crop_h = max(8, y1 - y0)
        if figsize is None:
            fig_w = max(3.0, crop_w / float(dpi) * 2.2)
            fig_h = max(3.0, crop_h / float(dpi) * 2.2)
        else:
            fig_w, fig_h = figsize

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_edgecolor("none")
        if overlay_on_raw and raw_stack is not None:
            raw = raw_stack[frame_to_index[frame_i]]
            raw_crop = raw[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
            if raw_crop.size == 0:
                raw_crop = np.zeros((crop_h, crop_w), dtype=float)
            raw_norm = _normalize_raw_image(
                raw_crop, contrast_limits=raw_contrast_limits, invert_image=invert_raw
            )
            cmap_now = base_cmap
            if "lut_whiten" in flash_modes and weight > 0:
                cmap_now = _whiten_cmap(base_cmap, lut_whiten_amount * weight)
            ax.imshow(raw_norm, cmap=cmap_now, origin="upper", zorder=0, interpolation="nearest")
            if "lut_whiten" in flash_modes and weight > 0:
                veil_h, veil_w = raw_norm.shape[:2]
                ax.imshow(
                    np.ones((max(veil_h, 1), max(veil_w, 1), 4))
                    * np.array([1.0, 1.0, 1.0, 0.22 * weight], dtype=float),
                    origin="upper",
                    interpolation="nearest",
                    zorder=1,
                )
            face = None
        else:
            face = canvas_rgb
            if "lut_whiten" in flash_modes and weight > 0:
                face = tuple(
                    np.clip(
                        np.array(canvas_rgb) * (1.0 - lut_whiten_amount * weight)
                        + lut_whiten_amount * weight,
                        0,
                        1,
                    )
                )
            ax.set_facecolor(face)
            fig.patch.set_facecolor(face)

        _draw_state_colored_trail(
            ax,
            vis,
            x0=x0,
            y0=y0,
            state_colors=state_colors,
            track_linewidth=track_linewidth,
            gold=gold,
            flash_weight=weight,
            draw_gold=("track_gold" in flash_modes),
            curve_samples=track_curve,
        )

        hx, hy = head_x - float(x0), head_y - float(y0)
        head_color = state_colors.get(str(shot["state"]), "0.85")
        radius = head_base_radius_px * (1.0 + head_pulse_scale * weight) if "head_pulse" in flash_modes else head_base_radius_px
        if "head_pulse" in flash_modes and weight > 0:
            ring = plt.Circle(
                (hx, hy),
                radius=radius * 1.55,
                facecolor=(*gold, 0.10 + 0.18 * weight),
                edgecolor=(*gold, 0.40 + 0.45 * weight),
                linewidth=2.0,
                zorder=5,
            )
            ax.add_patch(ring)
        ax.add_patch(
            plt.Circle(
                (hx, hy),
                radius=radius,
                facecolor=_as_rgba_tuple(head_color, alpha=1.0),
                edgecolor="none",
                linewidth=0,
                zorder=6,
            )
        )

        ax.set_xlim(0, crop_w)
        ax.set_ylim(crop_h, 0)
        ax.set_aspect("equal", adjustable="box")
        _hide_axes_chrome(ax)

        if show_timer:
            t_s = float(shot["time_s"]) if "time_s" in shot and pd.notna(shot["time_s"]) else frame_i * frame_interval
            ax.text(0.03, 0.96, f"{t_s:.2f} s", transform=ax.transAxes, color=text_color, fontsize=9, va="top")
        if show_state_label:
            label = str(shot["state"])
            if weight > 0 and pd.notna(shot.get("change_label")):
                label = str(shot["change_label"])
            ax.text(0.03, 0.04, label, transform=ax.transAxes, color=text_color, fontsize=9, va="bottom")
        if show_scale_bar:
            _add_scale_bar(ax, scale_bar_length_um, pixel_size_um, color=text_color)

        save_face = face if face is not None else "black"
        save_kwargs = dict(
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor=save_face,
            edgecolor="none",
        )
        if write_png:
            out_path = os.path.join(output_dir, f"{png_prefix}_f{frame_i:05d}_k{int(shot['output_index']):05d}.png")
            fig.savefig(out_path, **save_kwargs)
            png_paths.append(out_path)
        if write_gif:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", **save_kwargs)
            buf.seek(0)
            gif_frames.append(imageio.imread(buf))
        plt.close(fig)

    gif_path = None
    if write_gif and gif_frames:
        gif_path = os.path.join(output_dir, f"{png_prefix}.gif")
        try:
            imageio.mimsave(gif_path, gif_frames, fps=int(fps), loop=0)
        except Exception as exc:
            print(f"[state-movie] GIF write failed ({exc}).")
            gif_path = None

    preview = []
    n_out = len(png_paths) if png_paths else len(gif_frames)
    if preview_frames and n_out:
        change_rows = timeline.index[timeline["is_state_change"]].tolist()
        picks = [0, n_out // 2, n_out - 1]
        if change_rows:
            picks.insert(1, int(timeline.loc[change_rows[0], "output_index"]))
        picks = [i for i in sorted(set(picks)) if 0 <= i < n_out]
        if png_paths:
            preview = [png_paths[i] for i in picks]
        elif gif_path:
            preview = [gif_path]
        if show_plot:
            try:
                from IPython.display import Image, display

                for path in preview:
                    display(Image(filename=path))
            except Exception:
                pass

    result = {
        "output_dir": output_dir,
        "png_paths": png_paths,
        "gif_path": gif_path,
        "preview_paths": preview,
        "n_frames": int(len(track_df)),
        "n_pngs": int(len(png_paths)),
        "n_state_changes": int(track_df["is_state_change"].sum()),
        "unit_id": unit_id,
        "view_mode": view_mode,
        "state_col": used_state_col,
        "flash_modes": flash_modes,
        "raw_movie_path": resolved_raw_path,
        "overlay_on_raw": overlay_on_raw,
        "track_df": track_df,
        "timeline_df": timeline,
        "meta": meta_row,
        "state_colors": state_colors,
    }
    print(
        f"[state-movie] {view_mode} {unit_id}: "
        f"{len(png_paths)} PNGs, {len(gif_frames)} GIF frames, "
        f"{int(track_df['is_state_change'].sum())} state changes → {output_dir}"
    )
    return result


def make_state_change_track_movie(
    instant_df,
    superwindow_df=None,
    *,
    view_mode="superwindow",
    unit_id=None,
    state_col=None,
    track_id_col=None,
    target_state=None,
    min_target_fraction=0.0,
    max_target_fraction=1.0,
    require_states=None,
    exclude_states=None,
    min_n_distinct_states=1,
    max_n_distinct_states=None,
    min_n_switches=1,
    max_n_switches=None,
    min_steps=None,
    group_col=None,
    group_values=None,
    length=None,
    prefer="most_switches",
    n_select=1,
    random_state=0,
    exclude_ids=None,
    state_colors=None,
    state_order=None,
    **render_kwargs,
):
    """
    Filter tracks/superwindows by state composition, then render PNG movies.

    Filtering knobs live at the top; remaining keywords go to
    ``render_state_change_track_movie``.
    """
    summary = summarize_state_units_for_movie(
        superwindow_df=superwindow_df,
        instant_df=instant_df,
        view_mode=view_mode,
        state_col=state_col,
        track_id_col=track_id_col,
    )
    if unit_id is not None:
        picked = summary.loc[summary["unit_id"].astype(str) == str(unit_id)].copy()
        if picked.empty:
            raise ValueError(f"unit_id {unit_id!r} was not found in the {view_mode} summary")
        filtered = {"summary_df": summary, "selected_df": picked, "selected_ids": [picked.iloc[0]["unit_id"]]}
    else:
        filtered = filter_state_units_for_movie(
            summary,
            target_state=target_state,
            min_target_fraction=min_target_fraction,
            max_target_fraction=max_target_fraction,
            require_states=require_states,
            exclude_states=exclude_states,
            min_n_distinct_states=min_n_distinct_states,
            max_n_distinct_states=max_n_distinct_states,
            min_n_switches=min_n_switches,
            max_n_switches=max_n_switches,
            min_steps=min_steps,
            group_col=group_col,
            group_values=group_values,
            length=length,
            prefer=prefer,
            n_select=n_select,
            random_state=random_state,
            exclude_ids=exclude_ids,
        )

    movies = []
    base_output_dir = render_kwargs.pop("output_dir", None)
    for chosen_id in filtered["selected_ids"]:
        this_dir = None
        if base_output_dir is not None:
            this_dir = os.path.join(str(base_output_dir), str(chosen_id).replace(os.sep, "_"))
        movies.append(
            render_state_change_track_movie(
                instant_df,
                chosen_id,
                view_mode=view_mode,
                superwindow_df=superwindow_df,
                state_col=state_col,
                track_id_col=track_id_col,
                state_colors=state_colors,
                state_order=state_order,
                output_dir=this_dir,
                **render_kwargs,
            )
        )
    return {
        "summary_df": filtered["summary_df"],
        "selected_df": filtered["selected_df"],
        "selected_ids": filtered["selected_ids"],
        "movies": movies,
    }
