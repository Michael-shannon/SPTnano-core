"""Superwindows: runs of L consecutive transformer windows from one track, and their state sequences."""

import numpy as np
import pandas as pd
import polars as pl
import warnings
from pathlib import Path
from typing import Any, List, Sequence, Tuple
from typing import Mapping


def create_fixed_length_superwindows(
    windowed_df,
    superwindow_length=10,
    state_col='final_population',
    unique_id_col='unique_id',
    time_col='time_window',
    window_uid_col='window_uid',
    additional_cols=None,
    min_superwindows=1
):
    """
    Create fixed-length superwindows from longer tracks.
    
    Terminology:
    - Segment: Adjacent frames in instant_df
    - Window: Time window in windowed_df (made from segments)
    - Superwindow: Collection of consecutive windows (this function creates these)
    
    This function divides each track into non-overlapping superwindows of a fixed length.
    For example, with superwindow_length=10:
    - Superwindow 0: windows 0-9
    - Superwindow 1: windows 10-19
    - Superwindow 2: windows 20-29
    - etc.
    
    This ensures all superwindows have the same temporal coverage for fair comparison.
    
    Parameters
    ----------
    windowed_df : pl.DataFrame or pd.DataFrame
        Windowed dataframe with state assignments
    superwindow_length : int, default=10
        Number of windows per superwindow
    state_col : str, default='final_population'
        Column containing state classifications
    unique_id_col : str, default='unique_id'
        Column containing unique track IDs
    time_col : str, default='time_window'
        Column containing time window index
    window_uid_col : str, default='window_uid'
        Column containing unique window identifiers
    additional_cols : list, optional
        Additional metadata columns to include. Names not present in ``windowed_df``
        are skipped with a ``UserWarning``.
    min_superwindows : int, default=1
        Minimum number of complete superwindows required to include a track
        
    Returns
    -------
    pl.DataFrame or pd.DataFrame
        Dataframe with columns:
        - superwindow_id: Unique identifier for each superwindow (original_id + "_sw" + number)
        - original_track_id: Original track identifier
        - superwindow_number: Which superwindow this is (0, 1, 2, ...)
        - state_sequence: List of states in this superwindow
        - window_uids: List of window_uid values in this superwindow
        - n_windows: Number of windows (always = superwindow_length)
        - [additional_cols]: Metadata from original track
    """
    is_polars = isinstance(windowed_df, pl.DataFrame)
    
    if is_polars:
        df = windowed_df.to_pandas()
        return_polars = True
    else:
        df = windowed_df.copy()
        return_polars = False
    
    # Sort by track and time
    df = df.sort_values([unique_id_col, time_col])

    if additional_cols:
        missing = [c for c in additional_cols if c not in df.columns]
        if missing:
            warnings.warn(
                "create_fixed_length_superwindows: ignoring additional_cols not in "
                f"windowed_df: {missing}",
                UserWarning,
                stacklevel=2,
            )
        additional_cols = [c for c in additional_cols if c in df.columns]
    
    superwindows = []
    
    for track_id, track_data in df.groupby(unique_id_col):
        states = track_data[state_col].tolist()
        window_uids = track_data[window_uid_col].tolist()
        n_windows = len(states)
        
        # Calculate number of complete superwindows
        n_complete_superwindows = n_windows // superwindow_length
        
        # Skip if not enough superwindows
        if n_complete_superwindows < min_superwindows:
            continue
        
        # Extract metadata
        metadata = {}
        if additional_cols:
            for col in additional_cols:
                metadata[col] = track_data[col].iloc[0]
        
        # Create superwindows
        for sw_num in range(n_complete_superwindows):
            start_idx = sw_num * superwindow_length
            end_idx = start_idx + superwindow_length
            
            superwindow_states = states[start_idx:end_idx]
            superwindow_window_uids = window_uids[start_idx:end_idx]
            
            superwindow_record = {
                'superwindow_id': f"{track_id}_sw{sw_num}",
                'original_track_id': track_id,
                'superwindow_number': sw_num,
                'state_sequence': superwindow_states,
                'window_uids': superwindow_window_uids,
                'n_windows': len(superwindow_states)
            }
            
            # Add metadata
            superwindow_record.update(metadata)
            
            superwindows.append(superwindow_record)
    
    # Create dataframe
    superwindow_df = pd.DataFrame(superwindows)
    
    # Print summary
    n_original_tracks = df[unique_id_col].nunique()
    n_superwindows = len(superwindow_df)
    n_tracks_used = superwindow_df['original_track_id'].nunique()
    
    print(f"\n{'='*80}")
    print(f"FIXED-LENGTH SUPERWINDOW CREATION")
    print(f"{'='*80}")
    print(f"Superwindow length: {superwindow_length} windows")
    print(f"Original tracks: {n_original_tracks:,}")
    print(f"Tracks with ≥{min_superwindows} superwindow(s): {n_tracks_used:,}")
    print(f"Total superwindows created: {n_superwindows:,}")
    print(f"Avg superwindows per track: {n_superwindows / n_tracks_used:.1f}")
    
    # Show distribution of superwindows per track
    sws_per_track = superwindow_df.groupby('original_track_id').size()
    print(f"\nSuperwindows per track:")
    print(f"  Min: {sws_per_track.min()}")
    print(f"  Max: {sws_per_track.max()}")
    print(f"  Mean: {sws_per_track.mean():.1f}")
    print(f"  Median: {sws_per_track.median():.0f}")
    print(f"{'='*80}\n")
    
    # Convert back to polars if needed
    if return_polars:
        # Build the dataframe in Polars-native way to preserve List types
        # This allows native Polars operations like explode() on window_uids
        
        # Separate list columns from scalar columns
        list_cols = ['state_sequence', 'window_uids']
        scalar_cols = [c for c in superwindow_df.columns if c not in list_cols]
        
        # Convert scalar columns via pandas
        scalar_df = pl.from_pandas(superwindow_df[scalar_cols])
        
        # Add list columns as proper Polars List types
        superwindow_df = scalar_df.with_columns([
            pl.Series('state_sequence', superwindow_df['state_sequence'].tolist(), dtype=pl.List(pl.Utf8)),
            pl.Series('window_uids', superwindow_df['window_uids'].tolist(), dtype=pl.List(pl.Utf8)),
        ])
    
    return superwindow_df


def to_pandas(df_like: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Return a pandas copy from pandas or polars input."""
    return df_like.to_pandas() if isinstance(df_like, pl.DataFrame) else df_like.copy()


def apply_polars_filters(
    df: pl.DataFrame,
    filters: Mapping[str, Any],
    include_filenames: Sequence[str] | None = None,
    exclude_filenames: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Apply figuremaker-style simple filters to a Polars dataframe."""
    out = df
    for col, values in filters.items():
        if values is None or col not in out.columns:
            continue
        if callable(values):
            out = out.filter(values(pl.col(col)))
        elif isinstance(values, (list, tuple, set)):
            out = out.filter(pl.col(col).is_in(list(values)))
        else:
            out = out.filter(pl.col(col) == values)

    if include_filenames and "filename" in out.columns:
        out = out.filter(pl.col("filename").is_in(include_filenames))
    if exclude_filenames and "filename" in out.columns:
        out = out.filter(~pl.col("filename").is_in(exclude_filenames))
    return out


def add_derived_columns(
    df: pl.DataFrame,
    add_geno_nocodazole: bool = True,
    add_geno_mol: bool = True,
) -> pl.DataFrame:
    """Add common figuremaker derived columns when source columns are present."""
    out = df
    if add_geno_nocodazole and {"geno", "nocodazole_concentration"}.issubset(set(out.columns)):
        out = out.with_columns(
            (pl.col("geno").cast(pl.Utf8) + "_" + pl.col("nocodazole_concentration").cast(pl.Utf8)).alias(
                "geno_nocodazole"
            )
        )
    if add_geno_mol and {"geno", "mol"}.issubset(set(out.columns)):
        out = out.with_columns((pl.col("geno").cast(pl.Utf8) + "_" + pl.col("mol").cast(pl.Utf8)).alias("geno_mol"))
    return out


def filter_superwindows_by_window_uids(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    window_uids: Sequence[Any],
    require_all_windows: bool = True,
) -> pl.DataFrame:
    """Filter superwindows by whether their member window_uids are in a selected set."""
    swp = to_pandas(superwindow_df)
    keep_set = {str(uid) for uid in window_uids}
    if require_all_windows:
        keep = swp["window_uids"].apply(lambda uids: all(str(uid) in keep_set for uid in list(uids)))
    else:
        keep = swp["window_uids"].apply(lambda uids: any(str(uid) in keep_set for uid in list(uids)))
    return pl.from_pandas(swp.loc[keep].reset_index(drop=True))


def infer_state_order_from_superwindows(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    state_sequence_col: str = "state_sequence",
    preferred_order: Sequence[Any] | None = None,
) -> list[str]:
    """Infer observed states, preserving a preferred order when supplied."""
    swp = to_pandas(superwindow_df)
    observed: set[str] = set()
    for seq in swp[state_sequence_col]:
        observed.update(str(x) for x in list(seq))
    preferred = [str(s) for s in (preferred_order or []) if str(s) in observed]
    return preferred + sorted(observed - set(preferred))


def category_palette(categories: Sequence[Any], base_colors: Sequence[Any] | None = None, extra_palette: str = "tab20") -> dict[Any, Any]:
    """Build a deterministic category -> color mapping, extending base colors when needed."""
    import seaborn as sns

    categories = list(categories)
    colors = list(base_colors or [])
    if len(colors) < len(categories):
        for color in sns.color_palette(extra_palette, n_colors=max(len(categories) * 2, 20)).as_hex():
            if color not in colors:
                colors.append(color)
            if len(colors) >= len(categories):
                break
    return {cat: colors[i] for i, cat in enumerate(categories)}


def transition_probabilities_by_cluster(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    cluster_col: str = "superwindow_cluster_name",
    state_sequence_col: str = "state_sequence",
    state_order: Sequence[Any] | None = None,
    include_self: bool = True,
    mode: str = "mean_per_superwindow",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transition probabilities within each cluster using pooled or per-superwindow estimands."""
    df = to_pandas(superwindow_df)
    state_order = list(state_order or infer_state_order_from_superwindows(df, state_sequence_col))
    rows = []
    per_sw_rows = []
    state_to_idx = {str(s): i for i, s in enumerate(state_order)}
    for _, row in df.iterrows():
        seq = [str(x) for x in list(row[state_sequence_col])]
        cluster = row[cluster_col]
        sw_id = row.get("superwindow_id", np.nan)
        C = np.zeros((len(state_order), len(state_order)), dtype=float)
        for step, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
            if not include_self and a == b:
                continue
            if a not in state_to_idx or b not in state_to_idx:
                continue
            C[state_to_idx[a], state_to_idx[b]] += 1
            rows.append({"superwindow_id": sw_id, cluster_col: cluster, "transition_step": step, "from_state": a, "to_state": b, "transition_pair": f"{a} -> {b}", "is_self_transition": a == b})
        P = np.zeros_like(C)
        np.divide(C, C.sum(axis=1, keepdims=True), out=P, where=C.sum(axis=1, keepdims=True) > 0)
        for i, a in enumerate(state_order):
            for j, b in enumerate(state_order):
                per_sw_rows.append({"superwindow_id": sw_id, cluster_col: cluster, "from_state": a, "to_state": b, "transition_probability_per_superwindow": P[i, j], "n_transitions_in_superwindow": C[i, j]})

    events = pd.DataFrame(rows)
    per_sw = pd.DataFrame(per_sw_rows)
    pooled = events.groupby([cluster_col, "from_state", "to_state"], dropna=False).size().reset_index(name="n_transitions")
    idx = pd.MultiIndex.from_product([df[cluster_col].dropna().unique(), state_order, state_order], names=[cluster_col, "from_state", "to_state"])
    pooled = pooled.set_index([cluster_col, "from_state", "to_state"]).reindex(idx, fill_value=0).reset_index()
    pooled["n_outgoing_from_state"] = pooled.groupby([cluster_col, "from_state"])["n_transitions"].transform("sum")
    pooled["transition_probability_pooled"] = np.where(pooled["n_outgoing_from_state"] > 0, pooled["n_transitions"] / pooled["n_outgoing_from_state"], 0.0)
    agg = per_sw.groupby([cluster_col, "from_state", "to_state"], dropna=False).agg(
        transition_probability_mean_per_superwindow=("transition_probability_per_superwindow", "mean"),
        transition_probability_median_per_superwindow=("transition_probability_per_superwindow", "median"),
        n_superwindows=("superwindow_id", "nunique"),
    ).reset_index()
    out = pooled.merge(agg, on=[cluster_col, "from_state", "to_state"], how="left")
    out["transition_pair"] = out["from_state"] + " -> " + out["to_state"]
    out["is_self_transition"] = out["from_state"] == out["to_state"]
    col = {
        "pooled_counts": "transition_probability_pooled",
        "mean_per_superwindow": "transition_probability_mean_per_superwindow",
        "median_per_superwindow": "transition_probability_median_per_superwindow",
    }[mode]
    out["transition_probability"] = out[col]
    out["probability_mode"] = mode
    return events, out


def build_window_state_map_from_superwindows(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    superwindow_ids: Sequence[Any],
    state_sequence_col: str = "state_sequence",
    window_uids_col: str = "window_uids",
    cluster_col: str = "superwindow_cluster_name",
    group_col: str | None = None,
    visual_state_col: str = "superwindow_window_state_for_visualization",
) -> pd.DataFrame:
    """Expand selected superwindows into window_uid -> state rows."""
    swp = to_pandas(superwindow_df)
    swp["superwindow_id_str"] = swp["superwindow_id"].astype(str)
    rows = []
    for order_idx, sw_id in enumerate([sid for sid in superwindow_ids if sid is not None]):
        sub = swp.loc[swp["superwindow_id_str"] == str(sw_id)]
        if sub.empty:
            print(f"Warning: superwindow_id={sw_id} was not found in the superwindow dataframe")
            continue
        row = sub.iloc[0]
        window_uids = list(row[window_uids_col])
        states = [str(x) for x in list(row[state_sequence_col])]
        if len(window_uids) != len(states):
            print(
                f"Warning: superwindow_id={sw_id} has {len(window_uids)} window_uids "
                f"but {len(states)} states. Using the overlapping part."
            )
        for pos, (uid, state) in enumerate(zip(window_uids, states)):
            item = {
                "window_uid_str": str(uid),
                "window_uid": uid,
                "superwindow_id": row["superwindow_id"],
                "superwindow_cluster": row.get("superwindow_cluster", np.nan),
                cluster_col: row.get(cluster_col, np.nan),
                "plot_order_index": order_idx,
                "position_in_superwindow": pos,
                visual_state_col: state,
            }
            if group_col is not None:
                item[group_col] = row.get(group_col, np.nan)
            rows.append(item)
    if not rows:
        raise ValueError("No window/state rows were created from selected superwindows.")
    return pd.DataFrame(rows)


def map_window_states_to_instant(
    instant_df: pd.DataFrame | pl.DataFrame,
    window_state_map_df: pd.DataFrame,
    visual_state_col: str = "superwindow_window_state_for_visualization",
    group_col: str | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Filter instant rows to selected window_uids and map superwindow states onto them."""
    if isinstance(instant_df, pl.DataFrame) and "window_uid" in instant_df.columns:
        # Filter before converting: the full instant table is too big to hold as pandas.
        selected_uids = pl.Series(window_state_map_df["window_uid_str"].astype(str).unique())
        instant_df = instant_df.filter(pl.col("window_uid").cast(pl.Utf8).is_in(selected_uids))
    instant = to_pandas(instant_df)
    if "window_uid" not in instant.columns:
        raise KeyError("instant_df must contain 'window_uid'.")
    instant["window_uid_str"] = instant["window_uid"].astype(str)
    overwrite_cols = [
        "superwindow_id",
        "superwindow_cluster",
        "superwindow_cluster_name",
        "plot_order_index",
        "position_in_superwindow",
        visual_state_col,
    ]
    if group_col is not None:
        overwrite_cols.append(group_col)
    instant = instant.drop(columns=[col for col in overwrite_cols if col in instant.columns], errors="ignore")
    mapped = instant.merge(window_state_map_df, on="window_uid_str", how="inner")
    if mapped.empty:
        raise ValueError("No instant rows matched the selected superwindow window_uids.")
    if ("x_um" not in mapped.columns or "y_um" not in mapped.columns) and pixel_size_um is not None:
        if {"x", "y"}.issubset(mapped.columns):
            mapped["x_um"] = mapped["x"] * float(pixel_size_um)
            mapped["y_um"] = mapped["y"] * float(pixel_size_um)
    if "x_um" not in mapped.columns or "y_um" not in mapped.columns:
        raise KeyError("Need x_um/y_um, or x/y plus pixel_size_um, to plot tracks.")
    return mapped


def plot_state_colored_track_gallery(
    instant_df: pd.DataFrame | pl.DataFrame,
    superwindow_ids: Sequence[Any],
    state_col: str,
    state_colors: Mapping[Any, Any],
    cluster_col: str = "superwindow_cluster_name",
    cluster_colors: Mapping[Any, Any] | None = None,
    cluster_order: Sequence[Any] | None = None,
    ncols: int | None = None,
    subplot_width: float = 3.2,
    subplot_height: float = 3.2,
    same_xy_extent: bool = True,
    xy_span_um: float | None = 5.0,
    scale_bar_um: float = 1.0,
    text_color: str = "k",
    scale_bar_color: str = "k",
    transparent: bool = True,
    save_path: str | Path | None = None,
    show_superwindow_id: bool = True,
    show_cluster_in_title: bool = True,
):
    """Compact track gallery with segment colors from a mapped state column."""
    import matplotlib.pyplot as plt

    df = to_pandas(instant_df)
    df["superwindow_id_str"] = df["superwindow_id"].astype(str)
    ordered_ids = [sid for sid in superwindow_ids if sid is not None]
    tracks = []
    for sw_id in ordered_ids:
        sub = df.loc[df["superwindow_id_str"] == str(sw_id)].copy()
        if sub.empty:
            print(f"Warning: no instant rows for superwindow_id={sw_id}")
            continue
        cluster_name = sub[cluster_col].dropna().iloc[0] if cluster_col in sub.columns and sub[cluster_col].notna().any() else ""
        tracks.append((sw_id, cluster_name, sub))
    if not tracks:
        raise ValueError("No tracks to plot.")

    def extent(track: pd.DataFrame) -> float:
        return max(float(track["x_um"].max() - track["x_um"].min()), float(track["y_um"].max() - track["y_um"].min()))

    if same_xy_extent:
        span = float(xy_span_um) if xy_span_um is not None else max(max(extent(t[2]) for t in tracks), scale_bar_um * 1.5)
    else:
        span = None

    n_tracks = len(tracks)
    ncols = n_tracks if ncols is None else max(1, min(int(ncols), n_tracks))
    nrows = int(np.ceil(n_tracks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, subplot_height * nrows), squeeze=False)

    for ax, (sw_id, cluster_name, track) in zip(axes.ravel(), tracks):
        sort_cols = [col for col in ["position_in_superwindow", "frame"] if col in track.columns]
        track = track.sort_values(sort_cols) if sort_cols else track
        x = track["x_um"].to_numpy(dtype=float)
        y = track["y_um"].to_numpy(dtype=float)
        states = track[state_col].astype(str).to_numpy()
        for i in range(max(0, len(track) - 1)):
            ax.plot(x[i:i + 2], y[i:i + 2], color=state_colors.get(states[i], "lightgray"), linewidth=1.7, solid_capstyle="round")
        ax.scatter(x, y, s=4, color=[state_colors.get(s, "lightgray") for s in states], edgecolor="none", zorder=4)

        if span is not None:
            xm = float((track["x_um"].min() + track["x_um"].max()) / 2)
            ym = float((track["y_um"].min() + track["y_um"].max()) / 2)
            xlim = (xm - span / 2, xm + span / 2)
            ylim = (ym - span / 2, ym + span / 2)
        else:
            pad = 0.08 * max(extent(track), scale_bar_um)
            xlim = (float(track["x_um"].min() - pad), float(track["x_um"].max() + pad))
            ylim = (float(track["y_um"].min() - pad), float(track["y_um"].max() + pad))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color((cluster_colors or {}).get(cluster_name, "black"))
            spine.set_linewidth(2.4)
        title_bits = []
        if show_cluster_in_title and cluster_name not in ("", None):
            title_bits.append(str(cluster_name))
        if show_superwindow_id:
            title_bits.append(f"sw={sw_id}")
        ax.set_title("\n".join(title_bits) if title_bits else "", fontsize=8, color=text_color, pad=5)
        x0 = xlim[0] + 0.08 * (xlim[1] - xlim[0])
        y0 = ylim[0] + 0.08 * (ylim[1] - ylim[0])
        ax.plot([x0, x0 + scale_bar_um], [y0, y0], color=scale_bar_color, linewidth=2.2, solid_capstyle="butt")
        ax.text(x0 + scale_bar_um / 2, y0 + 0.035 * (ylim[1] - ylim[0]), f"{scale_bar_um:g} um", ha="center", va="bottom", color=scale_bar_color, fontsize=7)

    for ax in axes.ravel()[len(tracks):]:
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.86, wspace=0.10, hspace=0.28)
    handles = [plt.Line2D([0], [0], color=state_colors[state], linewidth=3, label=state) for state in state_colors]
    fig.legend(handles=handles, title="State", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=max(1, len(handles)), frameon=False, labelcolor=text_color)
    if save_path is not None:
        fig.savefig(save_path, format=Path(save_path).suffix.lstrip(".") or "svg", transparent=transparent, bbox_inches="tight")
    return fig, axes


def run_lengths_from_sequence(states: Sequence[Any]) -> List[Tuple[Any, int]]:
    """
    Split a state sequence into consecutive runs (state, length in windows).

    Parameters
    ----------
    states
        Ordered labels, one per window within a superwindow.

    Returns
    -------
    list of (state, run_length)
    """
    if not states:
        return []
    runs: List[Tuple[Any, int]] = []
    cur = states[0]
    ln = 1
    for s in states[1:]:
        if s == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur = s
            ln = 1
    runs.append((cur, ln))
    return runs


def _as_pandas_df(df_like):
    """Return a pandas DataFrame copy from pandas or polars input."""
    if isinstance(df_like, pl.DataFrame):
        return df_like.to_pandas()
    return df_like.copy()


def state_composition_fractions_from_sequences(
    superwindow_df,
    state_order,
    state_sequence_col="state_sequence",
    group_col=None,
):
    """
    Fraction of window-states occupied by each state, optionally within groups.

    Returns
    -------
    pd.DataFrame
        Columns: [group_col or "_group"], state, n_windows, fraction
    """
    df = _as_pandas_df(superwindow_df)
    state_order = list(state_order)
    group_name = group_col if group_col is not None else "_group"
    if group_col is None:
        work = df.copy()
        work[group_name] = "all"
    else:
        work = df.copy()

    rows = []
    for group_value, sub in work.groupby(group_name, dropna=False):
        counts = {s: 0 for s in state_order}
        total = 0
        for seq in sub[state_sequence_col]:
            for state in list(seq):
                state = str(state)
                if state in counts:
                    counts[state] += 1
                    total += 1
        for state in state_order:
            n = counts[state]
            rows.append(
                {
                    group_name: group_value,
                    "state": state,
                    "n_windows": n,
                    "fraction": (n / total) if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _transition_prob_lookup(transition_df, group_col, group_value, probability_col="transition_probability"):
    """Build {(from_state, to_state): probability} for one group."""
    sub = transition_df.loc[transition_df[group_col] == group_value]
    lookup = {}
    for _, row in sub.iterrows():
        lookup[(str(row["from_state"]), str(row["to_state"]))] = float(row[probability_col])
    return lookup


def _scale_arrow_style(prob, max_prob, min_lw=0.4, max_lw=4.5, min_ms=6.0, max_ms=18.0):
    """Map probability to linewidth and FancyArrowPatch mutation_scale."""
    if max_prob <= 0 or not np.isfinite(prob) or prob <= 0:
        return 0.0, 0.0
    t = float(np.clip(prob / max_prob, 0.0, 1.0))
    # Slightly emphasize mid/high probs for readability.
    t = t ** 0.75
    lw = min_lw + t * (max_lw - min_lw)
    ms = min_ms + t * (max_ms - min_ms)
    return lw, ms
