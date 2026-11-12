"""State-switching (plasticity) between behaviour states inside superwindows: tables, fans and heatmaps."""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import superwindows
from .superwindows import create_fixed_length_superwindows


def filter_out_filenames(
    df: pd.DataFrame | pl.DataFrame | None,
    filenames: Sequence[str],
) -> pd.DataFrame | pl.DataFrame | None:
    """Drop rows whose ``filename`` is in ``filenames`` (pandas or polars)."""
    if df is None or "filename" not in getattr(df, "columns", []):
        return df
    filenames = [str(x) for x in filenames]
    if isinstance(df, pl.DataFrame):
        return df.filter(~pl.col("filename").cast(pl.Utf8).is_in(filenames))
    out = df.copy()
    return out.loc[~out["filename"].astype(str).isin(filenames)].reset_index(drop=True)


def first_present_column(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    """Return the first candidate present in ``columns``."""
    cols = set(columns)
    for col in candidates:
        if col in cols:
            return col
    raise KeyError(f"Could not find a {label} column. Tried: {list(candidates)}")


def ensure_window_uid(df: pl.DataFrame, window_uid_col: str = "window_uid") -> pl.DataFrame:
    """Ensure a window_uid column exists (create row-index ids if missing)."""
    if window_uid_col in df.columns:
        return df
    return df.with_row_index("_superwindow_row_index").with_columns(
        pl.concat_str([pl.lit("row_"), pl.col("_superwindow_row_index").cast(pl.Utf8)]).alias(window_uid_col)
    )


def carry_forward_intermediate_states(
    df: pd.DataFrame | pl.DataFrame,
    *,
    state_col: str,
    track_col: str,
    time_col: str,
    resolved_state_col: str = "superwindow_resolved_state",
    allowed_states: Sequence[str] = ("bound", "exploratory", "confined", "transport"),
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Carry the last pure state forward over intermediate/unresolved labels.

    Returns
    -------
    resolved_df, full_df_with_flags, diagnostics_df
    """
    pdf = superwindows.to_pandas(df)
    allowed = {str(s).lower(): str(s) for s in allowed_states}
    pdf = pdf.sort_values([track_col, time_col]).reset_index(drop=True)

    resolved: list[Any] = []
    carried: list[bool] = []
    for _, grp in pdf.groupby(track_col, sort=False):
        last_pure = None
        for state in grp[state_col].astype("string"):
            key = str(state).lower() if pd.notna(state) else "<missing>"
            if key in allowed:
                last_pure = allowed[key]
                resolved.append(last_pure)
                carried.append(False)
            elif last_pure is not None:
                resolved.append(last_pure)
                carried.append(True)
            else:
                resolved.append(pd.NA)
                carried.append(False)

    pdf[resolved_state_col] = resolved
    pdf["superwindow_state_carried_forward"] = carried
    pdf["superwindow_state_was_intermediate"] = ~pdf[state_col].astype("string").str.lower().isin(set(allowed))
    keep = pdf[resolved_state_col].notna()
    diagnostics = pl.DataFrame(
        {
            "metric": [
                "input_windows",
                "resolved_windows",
                "carried_forward_windows",
                "leading_intermediate_windows_dropped",
                "fraction_carried_forward",
                "fraction_dropped",
            ],
            "value": [
                float(len(pdf)),
                float(keep.sum()),
                float(sum(carried)),
                float((~keep).sum()),
                float(sum(carried) / len(pdf)) if len(pdf) else np.nan,
                float((~keep).mean()) if len(pdf) else np.nan,
            ],
        }
    )
    return pl.from_pandas(pdf.loc[keep].copy()), pl.from_pandas(pdf), diagnostics


def prepare_superwindow_build_source(
    windowed_df: pl.DataFrame,
    *,
    window_uid_col: str = "window_uid",
    state_col_candidates: Sequence[str] = (
        "transformer_behaviour_state",
        "transformer_behavior_state",
        "amalgam_behaviour_state",
        "final_population",
    ),
    time_col_candidates: Sequence[str] = (
        "time_window",
        "window_number",
        "window_index",
        "frame_start",
        "first_frame",
        "frame",
    ),
    track_col_candidates: Sequence[str] = ("unique_id", "track_id", "particle", "trajectory_id"),
    include_intermediate_carry_forward: bool = True,
    allowed_states: Sequence[str] = ("bound", "exploratory", "confined", "transport"),
    resolved_state_col: str = "superwindow_resolved_state",
) -> dict[str, Any]:
    """Resolve columns and optionally carry-forward intermediate states for SW build."""
    source = ensure_window_uid(windowed_df, window_uid_col)
    cols = list(source.columns)
    state_col = first_present_column(cols, state_col_candidates, "state")
    time_col = first_present_column(cols, time_col_candidates, "time/order")
    track_col = first_present_column(cols, track_col_candidates, "track id")
    diagnostics = None
    build_df = source
    build_state_col = state_col
    if include_intermediate_carry_forward:
        build_df, _, diagnostics = carry_forward_intermediate_states(
            source,
            state_col=state_col,
            track_col=track_col,
            time_col=time_col,
            resolved_state_col=resolved_state_col,
            allowed_states=allowed_states,
        )
        build_state_col = resolved_state_col
    metadata_cols = [
        c
        for c in [
            "mol",
            "type",
            "geno",
            "group",
            "condition",
            "filename",
            "file_id",
            "cell",
            "identifier",
            "replicate",
            "location",
            "cell_key",
            track_col,
        ]
        if c in build_df.columns
    ]
    return {
        "build_df": build_df,
        "state_col": state_col,
        "build_state_col": build_state_col,
        "time_col": time_col,
        "track_col": track_col,
        "window_uid_col": window_uid_col,
        "metadata_cols": metadata_cols,
        "diagnostics": diagnostics,
    }


def build_or_load_fixed_superwindows(
    build_df: pl.DataFrame | pd.DataFrame,
    *,
    save_path: str | Path,
    superwindow_length: int = 6,
    state_col: str,
    unique_id_col: str,
    time_col: str,
    window_uid_col: str = "window_uid",
    additional_cols: Sequence[str] | None = None,
    rebuild: bool = False,
    write: bool = True,
) -> pl.DataFrame:
    """Load cached fixed-length superwindows or build and optionally save them."""
    save_path = Path(save_path)
    if save_path.exists() and not rebuild:
        return pl.read_parquet(save_path)

    sw = create_fixed_length_superwindows(
        build_df,
        superwindow_length=superwindow_length,
        state_col=state_col,
        unique_id_col=unique_id_col,
        time_col=time_col,
        window_uid_col=window_uid_col,
        additional_cols=list(additional_cols or []),
        min_superwindows=1,
    )
    out = pl.from_pandas(sw) if isinstance(sw, pd.DataFrame) else sw
    if write:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(save_path)
    return out


def prepare_filtered_window_tables(
    windowed_df: pl.DataFrame,
    instant_df: pl.DataFrame,
    *,
    filters: Mapping[str, Any] | None = None,
    include_filenames: Sequence[str] | None = None,
    exclude_filenames: Sequence[str] | None = None,
    add_geno_nocodazole: bool = True,
    add_geno_mol: bool = True,
    drop_unresolved_states: bool = False,
    state_col_for_unresolved: str | None = None,
) -> dict[str, Any]:
    """
    Apply figuremaker-style filters to windowed + instant tables only (no superwindows).
    """
    filters = dict(filters or {})
    analysis_windowed = superwindows.apply_polars_filters(
        windowed_df,
        filters,
        include_filenames=include_filenames,
        exclude_filenames=exclude_filenames,
    )
    analysis_windowed = superwindows.add_derived_columns(
        analysis_windowed,
        add_geno_nocodazole=add_geno_nocodazole,
        add_geno_mol=add_geno_mol,
    )
    if drop_unresolved_states and state_col_for_unresolved and state_col_for_unresolved in analysis_windowed.columns:
        analysis_windowed = analysis_windowed.filter(
            ~pl.col(state_col_for_unresolved).cast(pl.Utf8).str.to_lowercase().str.contains("unresolved")
        )

    window_uids = (
        analysis_windowed.select("window_uid").unique().to_series().cast(pl.Utf8).to_list()
        if "window_uid" in analysis_windowed.columns
        else []
    )
    analysis_instant = instant_df
    if "window_uid" in instant_df.columns and window_uids:
        analysis_instant = instant_df.filter(pl.col("window_uid").cast(pl.Utf8).is_in(window_uids))
    analysis_instant = superwindows.add_derived_columns(
        analysis_instant,
        add_geno_nocodazole=add_geno_nocodazole,
        add_geno_mol=add_geno_mol,
    )
    return {
        "analysis_windowed_df": analysis_windowed,
        "analysis_instant_df": analysis_instant,
        "analysis_window_uids": window_uids,
        "n_windowed": analysis_windowed.height,
        "n_instant": analysis_instant.height,
    }


def prepare_filtered_analysis_tables(
    windowed_df: pl.DataFrame,
    instant_df: pl.DataFrame,
    superwindow_df_all: pl.DataFrame | pd.DataFrame | None = None,
    *,
    filters: Mapping[str, Any] | None = None,
    include_filenames: Sequence[str] | None = None,
    exclude_filenames: Sequence[str] | None = None,
    add_geno_nocodazole: bool = True,
    add_geno_mol: bool = True,
    require_all_superwindow_windows: bool = True,
    drop_unresolved_states: bool = False,
    state_col_for_unresolved: str | None = None,
) -> dict[str, Any]:
    """
    Apply figuremaker-style filters and subset instant + superwindow tables.

    Superwindows are kept if all (or any) member window_uids are in the filtered set.
    Pass ``superwindow_df_all=None`` to skip superwindow subsetting (window-only workflows).
    """
    base = prepare_filtered_window_tables(
        windowed_df,
        instant_df,
        filters=filters,
        include_filenames=include_filenames,
        exclude_filenames=exclude_filenames,
        add_geno_nocodazole=add_geno_nocodazole,
        add_geno_mol=add_geno_mol,
        drop_unresolved_states=drop_unresolved_states,
        state_col_for_unresolved=state_col_for_unresolved,
    )
    if superwindow_df_all is None:
        empty_sw = pl.DataFrame({"superwindow_id": []})
        return {
            **base,
            "analysis_superwindow_df": empty_sw,
            "n_superwindows": 0,
        }

    analysis_superwindow = superwindows.filter_superwindows_by_window_uids(
        superwindow_df_all,
        base["analysis_window_uids"],
        require_all_windows=require_all_superwindow_windows,
    )
    analysis_superwindow = superwindows.add_derived_columns(
        analysis_superwindow if isinstance(analysis_superwindow, pl.DataFrame) else pl.from_pandas(analysis_superwindow),
        add_geno_nocodazole=add_geno_nocodazole,
        add_geno_mol=add_geno_mol,
    )
    return {
        **base,
        "analysis_superwindow_df": analysis_superwindow,
        "n_superwindows": analysis_superwindow.height,
    }


def summarize_analysis_tables(
    windowed_df: pl.DataFrame,
    instant_df: pl.DataFrame | None = None,
    superwindow_df: pl.DataFrame | None = None,
    group_col: str = "geno",
) -> pd.DataFrame:
    """Compact counts table for inspecting a filtered analysis subset."""
    rows = []
    if group_col in windowed_df.columns:
        for g, sub in windowed_df.group_by(group_col, maintain_order=True):
            gval = g[0] if isinstance(g, tuple) else g
            n_files = sub.select("filename").n_unique() if "filename" in sub.columns else np.nan
            n_tracks = (
                sub.select("unique_id").n_unique()
                if "unique_id" in sub.columns
                else (sub.select("track_id").n_unique() if "track_id" in sub.columns else np.nan)
            )
            rows.append(
                {
                    group_col: gval,
                    "n_windows": sub.height,
                    "n_filenames": n_files,
                    "n_tracks": n_tracks,
                }
            )
    else:
        rows.append({"n_windows": windowed_df.height})
    summary = pd.DataFrame(rows)
    if instant_df is not None:
        summary.attrs["n_instant_total"] = instant_df.height
    if superwindow_df is not None:
        summary.attrs["n_superwindows_total"] = superwindow_df.height
    return summary


def sample_superwindows_by_group(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    group_col: str,
    n_per_group: int = 8,
    group_order: Sequence[Any] | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample up to ``n_per_group`` superwindows for each group (for galleries)."""
    df = superwindows.to_pandas(superwindow_df)
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} not in superwindow dataframe")
    order = list(group_order or pd.Series(df[group_col].dropna().unique()).sort_values())
    rng = np.random.default_rng(random_state)
    parts = []
    for g in order:
        sub = df.loc[df[group_col] == g]
        if sub.empty:
            continue
        take = min(int(n_per_group), len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        picked = sub.loc[idx].copy()
        picked["gallery_slot"] = np.arange(take)
        parts.append(picked)
    if not parts:
        raise ValueError("No superwindows available to sample.")
    return pd.concat(parts, ignore_index=True)


def plot_ordered_superwindow_sequences(
    sampled_superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    state_colors: Mapping[Any, Any],
    group_order: Sequence[Any] | None = None,
    n_per_group: int | None = None,
    sequence_col: str = "state_sequence",
    superwindow_id_col: str = "superwindow_id",
    slot_col: str = "gallery_slot",
    show_superwindow_id: bool = True,
    show_slot_titles: bool = True,
    show_row_labels: bool = True,
    marker_size: float = 160,
    x_tick_interval: int | None = None,
    figsize: tuple[float, float] | None = None,
    line_color: str = "k",
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """
    Sequence-switch panels for an **already ordered** sample.

    Layout matches the track gallery: rows = ``group_order``, columns = gallery slots
    (0 .. n_per_group-1). Same ``superwindow_id`` occupies the same (row, col) as in
    ``plot_sampled_group_track_gallery``.

    Parameters
    ----------
    show_superwindow_id
        Annotate each panel with ``superwindow_id``.
    show_slot_titles
        Column titles like ``slot 0``.
    show_row_labels
        Row ylabel from ``group_col`` value (e.g. geno or ``r0``).
    marker_size
        Scatter marker area for state blobs.
    x_tick_interval
        If set (e.g. 5), x ticks are integers at that spacing; otherwise default ticks.
    """
    import matplotlib.pyplot as plt

    df = superwindows.to_pandas(sampled_superwindow_df)
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} not in sampled dataframe")
    if sequence_col not in df.columns:
        raise KeyError(f"{sequence_col!r} not in sampled dataframe")
    if slot_col not in df.columns:
        raise KeyError(f"{slot_col!r} not in sampled dataframe; sample with sample_superwindows_by_group first")

    order = list(group_order or pd.Series(df[group_col].dropna().unique()).sort_values())
    if n_per_group is None:
        n_per_group = int(df[slot_col].max()) + 1 if len(df) else 1
    n_per_group = max(1, int(n_per_group))
    states = list(state_order)
    n_states = len(states)
    marker_size = float(marker_size)

    if figsize is None:
        figsize = (max(6.0, 2.4 * n_per_group), max(3.0, 2.2 * len(order)))

    fig, axes = plt.subplots(len(order), n_per_group, figsize=figsize, squeeze=False)
    plotted_ids: list[Any] = []

    for row, group_value in enumerate(order):
        sub = df.loc[df[group_col] == group_value]
        for col in range(n_per_group):
            ax = axes[row, col]
            slot_rows = sub.loc[sub[slot_col] == col]
            if slot_rows.empty:
                ax.axis("off")
                plotted_ids.append(None)
                continue
            item = slot_rows.iloc[0]
            sequence = item[sequence_col]
            if not isinstance(sequence, list):
                sequence = list(sequence)
            sw_id = item.get(superwindow_id_col, None)
            plotted_ids.append(sw_id)

            for t, state in enumerate(sequence):
                state = str(state)
                if state not in states:
                    continue
                y_pos = states.index(state)
                ax.scatter(
                    t,
                    y_pos,
                    s=marker_size,
                    c=[state_colors.get(state, "lightgray")],
                    edgecolors=line_color,
                    linewidths=1.2,
                    zorder=5,
                )
                if t < len(sequence) - 1:
                    nxt = str(sequence[t + 1])
                    if nxt in states:
                        ax.plot(
                            [t, t + 1],
                            [y_pos, states.index(nxt)],
                            color=line_color,
                            linewidth=1.4,
                            alpha=0.55,
                            zorder=3,
                        )

            ax.set_yticks(range(n_states))
            ax.set_yticklabels(states if col == 0 else [], fontsize=8)
            xmax = max(len(sequence) - 0.5, 0.5)
            ax.set_xlim(-0.5, xmax)
            ax.set_ylim(-0.5, n_states - 0.5)
            if x_tick_interval is not None and int(x_tick_interval) > 0:
                step = int(x_tick_interval)
                n_pos = max(len(sequence), 1)
                ticks = list(range(0, n_pos, step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
            else:
                from matplotlib.ticker import MaxNLocator

                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.2, axis="x")
            if show_row_labels and col == 0:
                ax.set_ylabel(str(group_value), fontsize=10, fontweight="bold", color=line_color)
            else:
                ax.set_ylabel("")
            if show_slot_titles and row == 0:
                ax.set_title(f"slot {col}", fontsize=9, color=line_color)
            else:
                ax.set_title("")
            if show_superwindow_id and sw_id is not None:
                ax.text(
                    0.5,
                    -0.18,
                    str(sw_id),
                    transform=ax.transAxes,
                    fontsize=6,
                    color="gray",
                    ha="center",
                    va="top",
                    style="italic",
                )

    fig.suptitle(
        title or f"State sequences (same IDs / order as track gallery)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
        color=line_color,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            format=Path(save_path).suffix.lstrip(".") or "svg",
            transparent=True,
            bbox_inches="tight",
        )
    return fig, axes, plotted_ids


def plot_sampled_group_track_gallery(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    instant_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    state_colors: Mapping[Any, Any],
    group_order: Sequence[Any] | None = None,
    group_colors: Mapping[Any, Any] | None = None,
    n_per_group: int = 8,
    random_state: int = 0,
    visual_state_col: str = "superwindow_window_state_for_visualization",
    pixel_size_um: float | None = None,
    xy_span_um: float = 5.0,
    save_path: str | Path | None = None,
    also_plot_sequences: bool = True,
    sequence_save_path: str | Path | None = None,
    sequence_figsize: tuple[float, float] | None = None,
    line_color: str = "k",
):
    """
    Many example tracks per group (e.g. geno), state-colored, no clustering required.

    When ``also_plot_sequences=True`` (default), also draws the matching state-sequence
    grid for the **same** sampled ``superwindow_id``s in the same row/column order.
    """
    sampled = sample_superwindows_by_group(
        superwindow_df,
        group_col=group_col,
        n_per_group=n_per_group,
        group_order=group_order,
        random_state=random_state,
    )
    # Stable plot order: group blocks, then slot within group
    if group_order is not None:
        sampled[group_col] = pd.Categorical(sampled[group_col], categories=list(group_order), ordered=True)
        sampled = sampled.sort_values([group_col, "gallery_slot"])
    sw_ids = sampled["superwindow_id"].tolist()

    fig_seq = axes_seq = seq_ids = None
    if also_plot_sequences:
        fig_seq, axes_seq, seq_ids = plot_ordered_superwindow_sequences(
            sampled,
            group_col=group_col,
            state_order=state_order,
            state_colors=state_colors,
            group_order=group_order,
            n_per_group=n_per_group,
            figsize=sequence_figsize,
            line_color=line_color,
            save_path=sequence_save_path,
        )

    window_state_map_df = superwindows.build_window_state_map_from_superwindows(
        superwindow_df,
        sw_ids,
        cluster_col=group_col,
        group_col=group_col,
        visual_state_col=visual_state_col,
    )
    gallery_instant = superwindows.map_window_states_to_instant(
        instant_df,
        window_state_map_df,
        visual_state_col=visual_state_col,
        group_col=group_col,
        pixel_size_um=pixel_size_um,
    )
    fig, axes = superwindows.plot_state_colored_track_gallery(
        gallery_instant,
        sw_ids,
        state_col=visual_state_col,
        state_colors=state_colors,
        cluster_col=group_col,
        cluster_colors=group_colors,
        cluster_order=group_order,
        ncols=n_per_group,
        xy_span_um=xy_span_um,
        save_path=save_path,
    )
    return {
        "fig": fig,
        "axes": axes,
        "fig_sequences": fig_seq,
        "axes_sequences": axes_seq,
        "sequence_superwindow_ids": seq_ids,
        "sampled_superwindow_df": sampled,
        "superwindow_ids": sw_ids,
        "gallery_instant_df": gallery_instant,
    }


def attach_switch_scope(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    group_col: str,
    scope_col: str = "_switch_scope",
) -> pd.DataFrame:
    """Copy superwindows and set ``scope_col`` from ``group_col``."""
    sw = superwindows.to_pandas(superwindow_df)
    if group_col not in sw.columns:
        raise KeyError(f"{group_col!r} not in superwindow dataframe")
    out = sw.copy()
    out[scope_col] = out[group_col]
    return out


SWITCH_QUANTITIES = {
    "conditional",  # P(to | from)
    "flux",         # occupancy(from) * P(to | from)
}


DISPLAY_MODES = {
    "absolute",
    "delta_pp_vs_control",
    "delta_pct_vs_control",
    "delta_pct_vs_control_global",
    "delta_pct_vs_control_stabilized",
}


def attach_occupancy_and_flux(
    transition_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any] | None = None,
    probability_col: str = "transition_probability",
    composition_state_col: str = "state",
    composition_fraction_col: str = "fraction",
) -> pd.DataFrame:
    """
    Add ``from_state_occupancy`` and ``flux`` to a transition table.

    ``flux = occupancy(from_state) * P(to|from)``.

    Interpreted as the share of all steps that are the transition from→to
    (including self-transitions if present). Rare states are down-weighted.
    """
    tr = transition_df.copy()
    comp = composition_df.copy()
    if group_col not in tr.columns:
        raise KeyError(f"{group_col!r} not in transition_df")
    if group_col not in comp.columns:
        raise KeyError(f"{group_col!r} not in composition_df")
    if probability_col not in tr.columns:
        raise KeyError(f"{probability_col!r} not in transition_df")

    occ = comp[[group_col, composition_state_col, composition_fraction_col]].rename(
        columns={
            composition_state_col: "from_state",
            composition_fraction_col: "from_state_occupancy",
        }
    )
    out = tr.merge(occ, on=[group_col, "from_state"], how="left")
    out["from_state_occupancy"] = out["from_state_occupancy"].fillna(0.0).astype(float)
    out["flux"] = out["from_state_occupancy"] * out[probability_col].astype(float)
    out["conditional_probability"] = out[probability_col].astype(float)
    if state_order is not None:
        # keep a stable categorical order if useful downstream
        out["from_state"] = pd.Categorical(out["from_state"], categories=list(state_order), ordered=True)
        out["to_state"] = pd.Categorical(out["to_state"], categories=list(state_order), ordered=True)
        out = out.sort_values([group_col, "from_state", "to_state"]).reset_index(drop=True)
        out["from_state"] = out["from_state"].astype(str)
        out["to_state"] = out["to_state"].astype(str)
    return out


def resolve_switch_quantity_col(switch_quantity: str) -> str:
    """Map quantity toggle to the column used as the analysis value."""
    if switch_quantity not in SWITCH_QUANTITIES:
        raise ValueError(
            f"Unknown switch_quantity={switch_quantity!r}; choose from {sorted(SWITCH_QUANTITIES)}"
        )
    return "flux" if switch_quantity == "flux" else "transition_probability"


def prepare_switch_quantity_table(
    transition_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    *,
    group_col: str,
    switch_quantity: str = "conditional",
    state_order: Sequence[Any] | None = None,
    probability_col: str = "transition_probability",
) -> tuple[pd.DataFrame, str]:
    """
    Return (table, quantity_col) ready for fans/heatmaps/display modes.

    - conditional → uses ``transition_probability`` (= P(to|from))
    - flux → attaches occupancy and uses ``flux`` (= π_from * P(to|from))
    """
    quantity_col = resolve_switch_quantity_col(switch_quantity)
    if switch_quantity == "conditional":
        out = transition_df.copy()
        if "conditional_probability" not in out.columns and probability_col in out.columns:
            out["conditional_probability"] = out[probability_col].astype(float)
        return out, quantity_col
    out = attach_occupancy_and_flux(
        transition_df,
        composition_df,
        group_col=group_col,
        state_order=state_order,
        probability_col=probability_col,
    )
    return out, quantity_col


def control_offdiag_scale(
    ctrl_lookup: Mapping[tuple[Any, Any], float],
    state_order: Sequence[Any],
    *,
    how: str = "mean",
    min_scale: float = 1e-6,
) -> float:
    """
    Shared scale from control *off-diagonal* transition probabilities.

    Used by ``delta_pct_vs_control_global`` so small per-cell control Ps do not
    explode the denominator. ``how`` is ``\"mean\"``, ``\"median\"``, or ``\"max\"``.
    """
    vals = [
        float(ctrl_lookup.get((a, b), 0.0))
        for a in state_order
        for b in state_order
        if a != b and float(ctrl_lookup.get((a, b), 0.0)) > 0
    ]
    if not vals:
        return float(min_scale)
    if how == "median":
        scale = float(np.median(vals))
    elif how == "max":
        scale = float(np.max(vals))
    elif how == "mean":
        scale = float(np.mean(vals))
    else:
        raise ValueError(f"Unknown global scale how={how!r}; use mean|median|max")
    return max(scale, float(min_scale))


def compute_transition_display_value(
    p: float,
    p_ctrl: float,
    *,
    group_value: Any,
    control_scope: Any,
    display_mode: str,
    global_scale: float | None = None,
    stabilize_eps: float | None = None,
    ctrl_eps: float = 1e-9,
) -> tuple[float, str]:
    """
    Map absolute P to a display value under ``display_mode``.

    Modes
    -----
    absolute
        raw P
    delta_pp_vs_control
        100*(P - P_ctrl)  [percentage points]
    delta_pct_vs_control
        100*(P - P_ctrl)/P_ctrl  [can blow up if P_ctrl is tiny]
    delta_pct_vs_control_global
        100*(P - P_ctrl)/global_scale
        where global_scale is typically the mean control off-diagonal P.
        Read as: change as a % of a *typical* control switch rate.
    delta_pct_vs_control_stabilized
        100*(P - P_ctrl)/(P_ctrl + eps)
        eps defaults to global_scale (same typical control switch rate),
        so rare control transitions are damped instead of exploding.
    """
    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown display_mode={display_mode!r}; choose from {sorted(DISPLAY_MODES)}")
    p = float(p)
    p_ctrl = float(p_ctrl)
    if display_mode == "absolute" or group_value == control_scope:
        return p, "absolute"
    if display_mode == "delta_pp_vs_control":
        return 100.0 * (p - p_ctrl), "delta_pp"
    if display_mode == "delta_pct_vs_control":
        if abs(p_ctrl) < ctrl_eps:
            return np.nan, "delta_pct"
        return 100.0 * (p - p_ctrl) / p_ctrl, "delta_pct"
    if display_mode == "delta_pct_vs_control_global":
        scale = float(global_scale) if global_scale is not None else 1.0
        if scale < ctrl_eps:
            return np.nan, "delta_pct_global"
        return 100.0 * (p - p_ctrl) / scale, "delta_pct_global"
    # stabilized relative %
    eps = float(stabilize_eps) if stabilize_eps is not None else (float(global_scale) if global_scale is not None else 0.05)
    denom = p_ctrl + max(eps, ctrl_eps)
    return 100.0 * (p - p_ctrl) / denom, "delta_pct_stab"


def build_display_transition_df(
    transition_df: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[Any],
    state_order: Sequence[Any],
    control_scope: str,
    display_mode: str = "delta_pct_vs_control",
    probability_col: str = "transition_probability",
    ctrl_eps: float = 1e-9,
    global_scale_stat: str = "mean",
    stabilize_eps: float | None = None,
) -> pd.DataFrame:
    """
    Long table of absolute P and display values (absolute / % / pp / global-% vs control).

    Control panel always keeps absolute P when ``display_mode`` is a delta mode.

    For ``delta_pct_vs_control_global`` / ``_stabilized``, ``global_scale_stat``
    chooses how the shared control off-diagonal scale is computed
    (``mean`` / ``median`` / ``max``). If ``stabilize_eps`` is None, stabilization
    uses that same global scale as ``eps``.
    """
    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown display_mode={display_mode!r}; choose from {sorted(DISPLAY_MODES)}")
    ctrl = superwindows._transition_prob_lookup(transition_df, group_col, control_scope, probability_col)
    global_scale = control_offdiag_scale(ctrl, state_order, how=global_scale_stat)
    rows = []
    for group_value in group_order:
        probs = superwindows._transition_prob_lookup(transition_df, group_col, group_value, probability_col)
        for a in state_order:
            for b in state_order:
                p = float(probs.get((a, b), 0.0))
                p_ctrl = float(ctrl.get((a, b), 0.0))
                val, kind = compute_transition_display_value(
                    p,
                    p_ctrl,
                    group_value=group_value,
                    control_scope=control_scope,
                    display_mode=display_mode,
                    global_scale=global_scale,
                    stabilize_eps=stabilize_eps,
                    ctrl_eps=ctrl_eps,
                )
                rows.append(
                    {
                        group_col: group_value,
                        "from_state": a,
                        "to_state": b,
                        "transition_probability": p,
                        "control_probability": p_ctrl,
                        "display_value": val,
                        "display_kind": kind,
                        "global_scale": global_scale,
                        "is_self_transition": a == b,
                    }
                )
    out = pd.DataFrame(rows)
    out.attrs["global_scale"] = global_scale
    out.attrs["global_scale_stat"] = global_scale_stat
    out.attrs["display_mode"] = display_mode
    return out


def _format_switch_label(val: Any, kind: str) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "n/a"
    if kind == "absolute":
        return f"{float(val):.2f}"
    return f"{float(val):+.0f}%"


def plot_focal_fan_switch_networks(
    transition_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[Any],
    state_order: Sequence[Any],
    state_colors: Mapping[Any, Any],
    control_scope: str,
    focal_state: str = "bound",
    display_mode: str = "delta_pct_vs_control",
    size_boxes_by_frequency: bool = True,
    show_zero_or_tiny_arrows: bool = True,
    min_arrow_prob: float = 0.01,
    line_color: str = "k",
    periph_x: float = 0.84,
    periph_curve: float = 0.42,
    periph_x_nudge: float = 0.06,
    box_w0: float = 0.20,
    box_h0: float = 0.10,
    box_min_scale: float = 0.55,
    box_max_scale: float = 1.35,
    global_scale_stat: str = "mean",
    stabilize_eps: float | None = None,
    probability_col: str = "transition_probability",
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """
    One fan network panel per group: spokes from ``focal_state`` + curved peripheral links.

    Arrow thickness uses absolute values from ``probability_col`` (conditional P or flux);
    labels follow ``display_mode`` (control stays absolute under delta modes).
    Box sizes optionally scale by within-group composition.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    import matplotlib.pyplot as plt

    if focal_state not in list(state_order):
        raise ValueError(f"focal_state={focal_state!r} not in state_order")
    if control_scope not in list(group_order):
        raise KeyError(f"control_scope={control_scope!r} not in group_order={list(group_order)}")
    if display_mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown display_mode={display_mode!r}; choose from {sorted(DISPLAY_MODES)}")
    if probability_col not in transition_df.columns:
        raise KeyError(f"{probability_col!r} not in transition_df columns")

    states = list(state_order)
    others = [s for s in states if s != focal_state]
    ctrl_probs = superwindows._transition_prob_lookup(transition_df, group_col, control_scope, probability_col)
    global_scale = control_offdiag_scale(ctrl_probs, states, how=global_scale_stat)
    n_genos = max(1, len(group_order))

    def frac_lookup(group_value):
        sub = composition_df.loc[composition_df[group_col] == group_value].set_index("state")
        fracs = {s: float(sub.loc[s, "fraction"]) if s in sub.index else 0.0 for s in states}
        return fracs, (max(fracs.values()) if fracs else 0.0)

    def box_scale(frac, max_frac):
        if (not size_boxes_by_frequency) or max_frac <= 0:
            return 1.0
        rel = float(frac) / float(max_frac)
        return box_min_scale + (box_max_scale - box_min_scale) * rel

    def display_value(p, p_ctrl, group_value):
        return compute_transition_display_value(
            p,
            p_ctrl,
            group_value=group_value,
            control_scope=control_scope,
            display_mode=display_mode,
            global_scale=global_scale,
            stabilize_eps=stabilize_eps,
        )

    def draw_box(ax, x, y, state, scale=1.0, fontsize=9):
        w, h = box_w0 * scale, box_h0 * scale
        fs = max(6.0, fontsize * (0.85 + 0.15 * scale))
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.5 * w, y - 0.5 * h),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=state_colors.get(state, "lightgray"),
                edgecolor=line_color,
                linewidth=1.0,
                alpha=0.95,
                zorder=5,
                clip_on=False,
            )
        )
        ax.text(x, y, str(state), ha="center", va="center", fontsize=fs, fontweight="bold", color=line_color, zorder=6, clip_on=False)

    def draw_bidir(
        ax,
        pos_a,
        pos_b,
        p_ab,
        p_ba,
        state_a,
        state_b,
        max_prob,
        group_value,
        pair_offset=0.045,
        shrink_a=0.10,
        shrink_b=0.10,
        curve_rad=0.0,
        label_push=1.2,
        x_nudge=0.0,
        force_show=False,
    ):
        x0, y0 = pos_a
        x1, y1 = pos_b
        x0, x1 = x0 + x_nudge, x1 + x_nudge
        vx, vy = x1 - x0, y1 - y0
        norm = max(np.hypot(vx, vy), 1e-9)
        px, py = -vy / norm, vx / norm
        for lane, p, src, x_s, y_s, x_e, y_e, rad_sign, a_from, a_to, sh_s, sh_e in (
            (-0.5, p_ab, state_a, x0, y0, x1, y1, +1.0, state_a, state_b, shrink_a, shrink_b),
            (+0.5, p_ba, state_b, x1, y1, x0, y0, -1.0, state_b, state_a, shrink_b, shrink_a),
        ):
            tiny = p < min_arrow_prob
            if tiny and not (force_show or show_zero_or_tiny_arrows):
                continue
            if max_prob > 0 and p > 0:
                lw, ms = superwindows._scale_arrow_style(p, max_prob)
            else:
                lw, ms = 0.5, 7.0
            if tiny:
                lw, ms = min(lw, 0.6), min(ms, 7.0)
                alpha, ls = 0.35, (0, (3, 2))
            else:
                alpha, ls = 0.9, "solid"
            ox, oy = lane * pair_offset * px, lane * pair_offset * py
            dx, dy = x_e - x_s, y_e - y_s
            nrm = max(np.hypot(dx, dy), 1e-9)
            uux, uuy = dx / nrm, dy / nrm
            sx0, sy0 = x_s + sh_s * uux + ox, y_s + sh_s * uuy + oy
            sx1, sy1 = x_e - sh_e * uux + ox, y_e - sh_e * uuy + oy
            rad = float(curve_rad) * rad_sign
            ax.add_patch(
                FancyArrowPatch(
                    (sx0, sy0),
                    (sx1, sy1),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    mutation_scale=max(ms, 7),
                    linewidth=max(lw, 0.5),
                    color=state_colors.get(src, line_color),
                    alpha=alpha,
                    linestyle=ls,
                    zorder=3,
                    clip_on=False,
                )
            )
            p_ctrl = ctrl_probs.get((a_from, a_to), 0.0)
            val, kind = display_value(p, p_ctrl, group_value)
            ddx, ddy = sx1 - sx0, sy1 - sy0
            mx = 0.5 * (sx0 + sx1) + 0.5 * rad * ddy + label_push * ox
            my = 0.5 * (sy0 + sy1) - 0.5 * rad * ddx + label_push * oy
            ax.text(
                mx,
                my,
                _format_switch_label(val, kind),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=line_color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
                zorder=6,
                clip_on=False,
            )

    def draw_fan(ax, probs, max_prob, group_value, fracs, max_frac):
        ax.set_xlim(-0.08, 1.20)
        ax.set_ylim(-0.10, 1.10)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        scales = {s: box_scale(fracs.get(s, 0.0), max_frac) for s in states}
        shrinks = {s: 0.08 + 0.10 * scales[s] for s in states}
        focal_pos = (0.18, 0.50)
        draw_box(ax, *focal_pos, focal_state, scale=scales[focal_state], fontsize=10)
        if not others:
            return
        ys = np.linspace(0.88, 0.12, len(others)) if len(others) > 1 else np.array([0.50])
        xs = periph_x - 0.04 * np.cos(np.linspace(-0.9, 0.9, len(others)))
        positions = {focal_state: focal_pos}
        for x, y, state in zip(xs, ys, others):
            positions[state] = (float(x), float(y))
            draw_box(ax, float(x), float(y), state, scale=scales[state], fontsize=9)
            draw_bidir(
                ax,
                focal_pos,
                (float(x), float(y)),
                probs.get((focal_state, state), 0.0),
                probs.get((state, focal_state), 0.0),
                focal_state,
                state,
                max_prob,
                group_value,
                pair_offset=0.030,
                shrink_a=shrinks[focal_state],
                shrink_b=shrinks[state],
                curve_rad=0.08,
            )
        for i, state_a in enumerate(others):
            for state_b in others[i + 1 :]:
                draw_bidir(
                    ax,
                    positions[state_a],
                    positions[state_b],
                    probs.get((state_a, state_b), 0.0),
                    probs.get((state_b, state_a), 0.0),
                    state_a,
                    state_b,
                    max_prob,
                    group_value,
                    pair_offset=0.028,
                    shrink_a=shrinks[state_a],
                    shrink_b=shrinks[state_b],
                    curve_rad=-abs(periph_curve),
                    label_push=1.6,
                    x_nudge=periph_x_nudge,
                    force_show=True,
                )

    fig, axes = plt.subplots(1, n_genos, figsize=(4.2 * n_genos, 4.8), squeeze=False)
    for ax in axes.ravel():
        ax.set_box_aspect(1)

    for col, group_value in enumerate(group_order):
        probs = superwindows._transition_prob_lookup(transition_df, group_col, group_value, probability_col)
        off = [probs.get((a, b), 0.0) for a in states for b in states if a != b]
        max_prob = max(off) if off else 0.0
        fracs, max_frac = frac_lookup(group_value)
        ax = axes[0, col]
        subtitle = str(group_value)
        if display_mode != "absolute" and group_value == control_scope:
            subtitle = f"{group_value}\n(control, absolute P)"
        elif display_mode != "absolute":
            subtitle = f"{group_value}\n(vs {control_scope})"
        ax.set_title(subtitle, fontsize=9, color=line_color, pad=6)
        draw_fan(ax, probs, max_prob, group_value, fracs, max_frac)

    if title is None:
        title = f"Fan from {focal_state} | display={display_mode}"
    fig.suptitle(title, color=line_color, fontsize=11, y=1.03)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format=Path(save_path).suffix.lstrip(".") or "svg", transparent=True, bbox_inches="tight")
    return fig, axes


def plot_transition_heatmaps_display(
    display_df: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[Any],
    state_order: Sequence[Any],
    control_scope: str,
    display_mode: str = "delta_pct_vs_control",
    absolute_cbar_label: str = "P(next | current)",
    absolute_vmax: float | None = None,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """Heatmaps of ``display_value``; diverging cmap for vs-control panels."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = max(1, len(group_order))
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows), squeeze=False)
    use_delta = display_mode != "absolute"
    vmax = 1.0
    abs_vals = display_df.loc[display_df["display_kind"] == "absolute", "display_value"]
    if abs_vals is None or len(abs_vals) == 0:
        abs_vals = display_df["display_value"]
    abs_nums = np.asarray(abs_vals.to_numpy(dtype=float), dtype=float)
    abs_nums = abs_nums[np.isfinite(abs_nums)]
    data_max = float(np.nanmax(abs_nums)) if len(abs_nums) else 1.0
    # Flux absolute values are often << 1; autoscale unless user overrides.
    vmax_abs = float(absolute_vmax) if absolute_vmax is not None else (1.0 if data_max > 0.5 else max(data_max, 1e-6))
    if use_delta:
        delta_vals = display_df.loc[
            (display_df[group_col] != control_scope) & display_df["display_value"].notna(),
            "display_value",
        ]
        vmax = float(np.nanmax(np.abs(delta_vals))) if len(delta_vals) else 1.0
        vmax = max(vmax, 1.0)

    for ax, group_value in zip(axes.ravel(), group_order):
        sub = display_df.loc[display_df[group_col] == group_value]
        mat = (
            sub.pivot_table(index="from_state", columns="to_state", values="display_value", aggfunc="mean")
            .reindex(index=list(state_order), columns=list(state_order))
        )
        is_ctrl = (group_value == control_scope) or (display_mode == "absolute")
        if is_ctrl:
            sns.heatmap(
                mat.fillna(0.0),
                vmin=0,
                vmax=vmax_abs,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": absolute_cbar_label},
                ax=ax,
            )
        else:
            cbar_label = {
                "delta_pp_vs_control": f"pp diff vs {control_scope}",
                "delta_pct_vs_control": f"% diff vs {control_scope} (per-cell)",
                "delta_pct_vs_control_global": f"% of typical control switch vs {control_scope}",
                "delta_pct_vs_control_stabilized": f"stabilized % diff vs {control_scope}",
            }.get(display_mode, f"diff vs {control_scope}")
            sns.heatmap(
                mat,
                vmin=-vmax,
                vmax=vmax,
                cmap="RdBu_r",
                center=0.0,
                annot=True,
                fmt=".0f",
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": cbar_label},
                ax=ax,
                mask=mat.isna(),
            )
        for i in range(len(state_order)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2))
        ttl = str(group_value)
        if use_delta and group_value == control_scope:
            ttl = f"{group_value} (control, absolute P)"
        elif use_delta:
            ttl = f"{group_value} (vs {control_scope})"
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title or f"Transition heatmaps | display={display_mode}", fontsize=11, y=1.02)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format=Path(save_path).suffix.lstrip(".") or "svg", transparent=True, bbox_inches="tight")
    return fig, axes


LENGTH_AGG_POLICIES = {
    "equal_length_mean",
    "n_superwindow_weighted",
    "transition_count_weighted",
    "track_equal_then_length_equal",
}


def build_or_load_fixed_superwindows_for_lengths(
    build_df: pl.DataFrame | pd.DataFrame,
    *,
    save_dir: str | Path,
    lengths: Sequence[int],
    state_col: str,
    unique_id_col: str,
    time_col: str,
    window_uid_col: str = "window_uid",
    additional_cols: Sequence[str] | None = None,
    rebuild: bool = False,
    write: bool = True,
    source_tag: str = "prefilter_carry_forward",
    length_col: str = "superwindow_length",
) -> pl.DataFrame:
    """
    Build/load fixed-length superwindows for each length and concatenate.

    Each length is cached separately under ``save_dir``. The concatenated
    table adds ``superwindow_length`` and remaps ``superwindow_id`` to
    ``L{length}_{original_id}`` so IDs remain unique across lengths.
    Existing single-length helpers are left unchanged.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted({int(L) for L in lengths if int(L) >= 2})
    if not lengths:
        raise ValueError("lengths must contain at least one integer >= 2")

    frames: list[pl.DataFrame] = []
    for L in lengths:
        path = save_dir / f"superwindow_df_all_{source_tag}_fixed{L}.parquet"
        sw = build_or_load_fixed_superwindows(
            build_df,
            save_path=path,
            superwindow_length=L,
            state_col=state_col,
            unique_id_col=unique_id_col,
            time_col=time_col,
            window_uid_col=window_uid_col,
            additional_cols=additional_cols,
            rebuild=rebuild,
            write=write,
        )
        if not isinstance(sw, pl.DataFrame):
            sw = pl.from_pandas(sw)
        sw = sw.with_columns(
            pl.lit(int(L)).alias(length_col),
            (pl.lit(f"L{L}_") + pl.col("superwindow_id").cast(pl.Utf8)).alias("superwindow_id"),
        )
        frames.append(sw)
        print(f"length={L:3d}  n_superwindows={sw.height:6d}  cache={path.name}")

    out = pl.concat(frames, how="diagonal_relaxed")
    print(f"concatenated multi-length table: {out.shape}  lengths={lengths}")
    return out


def subset_superwindows_by_length(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    length: int,
    length_col: str = "superwindow_length",
) -> pd.DataFrame:
    """Return only superwindows of the requested length (pandas)."""
    df = superwindows.to_pandas(superwindow_df)
    if length_col not in df.columns:
        raise KeyError(f"{length_col!r} missing; build with multi-length helpers first")
    return df.loc[df[length_col].astype(int) == int(length)].copy()


def transition_probabilities_by_length(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any] | None = None,
    mode: str = "mean_per_superwindow",
    lengths: Sequence[int] | None = None,
    length_col: str = "superwindow_length",
    include_self: bool = True,
) -> pd.DataFrame:
    """
    Compute transition probabilities separately for each superwindow length.

    Returns a long table with the usual transition columns plus ``superwindow_length``.
    """
    df = superwindows.to_pandas(superwindow_df)
    if length_col not in df.columns:
        raise KeyError(f"{length_col!r} missing")
    if group_col not in df.columns:
        raise KeyError(f"{group_col!r} missing")
    length_vals = sorted(int(x) for x in df[length_col].dropna().unique())
    if lengths is not None:
        want = {int(L) for L in lengths}
        length_vals = [L for L in length_vals if L in want]
    state_order = list(state_order or superwindows.infer_state_order_from_superwindows(df))

    pieces = []
    for L in length_vals:
        sub = df.loc[df[length_col].astype(int) == L]
        if sub.empty:
            continue
        _, prob = superwindows.transition_probabilities_by_cluster(
            sub,
            cluster_col=group_col,
            state_order=state_order,
            mode=mode,
            include_self=include_self,
        )
        prob = prob.copy()
        prob[length_col] = int(L)
        pieces.append(prob)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["probability_mode"] = mode
    return out


def _weighted_mean_rows(
    sub: pd.DataFrame,
    value_col: str,
    weight_col: str | None,
) -> float:
    vals = pd.to_numeric(sub[value_col], errors="coerce")
    if weight_col is None:
        return float(vals.mean()) if len(vals) else np.nan
    w = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0.0)
    if float(w.sum()) <= 0:
        return float(vals.mean()) if len(vals) else np.nan
    return float(np.average(vals, weights=w))


def aggregate_transitions_across_lengths(
    per_length_prob_df: pd.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    policy: str = "equal_length_mean",
    lengths: Sequence[int] | None = None,
    length_col: str = "superwindow_length",
    probability_col: str = "transition_probability",
) -> pd.DataFrame:
    """
    Collapse a per-length transition table into one genotype-level matrix.

    Policies
    --------
    equal_length_mean
        Each length contributes equally (recommended default for amalgams).
    n_superwindow_weighted
        Weight each length by ``n_superwindows`` (favours short L).
    transition_count_weighted
        Weight by ``n_transitions`` / outgoing counts when present; else falls
        back to ``n_superwindows``.

    Returns a table shaped like ``transition_probabilities_by_cluster`` output
    so fan / heatmap helpers can be reused unchanged.
    """
    if policy not in {"equal_length_mean", "n_superwindow_weighted", "transition_count_weighted"}:
        raise ValueError(
            f"Unknown policy={policy!r}; for track-aware use "
            "aggregate_transitions_track_aware_across_lengths"
        )
    df = per_length_prob_df.copy()
    if lengths is not None:
        df = df.loc[df[length_col].astype(int).isin({int(L) for L in lengths})].copy()
    if df.empty:
        return df

    weight_col = None
    if policy == "n_superwindow_weighted":
        weight_col = "n_superwindows" if "n_superwindows" in df.columns else None
    elif policy == "transition_count_weighted":
        if "n_transitions" in df.columns:
            weight_col = "n_transitions"
        elif "n_outgoing_from_state" in df.columns:
            weight_col = "n_outgoing_from_state"
        elif "n_superwindows" in df.columns:
            weight_col = "n_superwindows"

    rows = []
    keys = [group_col, "from_state", "to_state"]
    for key, sub in df.groupby(keys, dropna=False):
        g, a, b = key
        p = _weighted_mean_rows(sub, probability_col, weight_col)
        row = {
            group_col: g,
            "from_state": a,
            "to_state": b,
            "transition_pair": f"{a} -> {b}",
            "is_self_transition": str(a) == str(b),
            "transition_probability": p,
            "probability_mode": f"multi_length:{policy}",
            "n_lengths": int(sub[length_col].nunique()),
            "length_dispersion": float(pd.to_numeric(sub[probability_col], errors="coerce").std(ddof=0)),
        }
        if "n_superwindows" in sub.columns:
            row["n_superwindows"] = int(pd.to_numeric(sub["n_superwindows"], errors="coerce").fillna(0).sum())
        if "n_transitions" in sub.columns:
            row["n_transitions"] = float(pd.to_numeric(sub["n_transitions"], errors="coerce").fillna(0).sum())
        rows.append(row)

    out = pd.DataFrame(rows)
    out["from_state"] = pd.Categorical(out["from_state"], categories=list(state_order), ordered=True)
    out["to_state"] = pd.Categorical(out["to_state"], categories=list(state_order), ordered=True)
    out = out.sort_values([group_col, "from_state", "to_state"]).reset_index(drop=True)
    out["from_state"] = out["from_state"].astype(str)
    out["to_state"] = out["to_state"].astype(str)
    return out


def _per_superwindow_transition_rows(
    superwindow_df: pd.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    state_sequence_col: str = "state_sequence",
    track_col: str = "original_track_id",
    length_col: str = "superwindow_length",
) -> pd.DataFrame:
    """Row-normalized P(to|from) for every superwindow (including zeros)."""
    df = superwindows.to_pandas(superwindow_df)
    state_to_idx = {str(s): i for i, s in enumerate(state_order)}
    rows = []
    for _, row in df.iterrows():
        seq = [str(x) for x in list(row[state_sequence_col])]
        C = np.zeros((len(state_order), len(state_order)), dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            if a in state_to_idx and b in state_to_idx:
                C[state_to_idx[a], state_to_idx[b]] += 1
        P = np.zeros_like(C)
        np.divide(C, C.sum(axis=1, keepdims=True), out=P, where=C.sum(axis=1, keepdims=True) > 0)
        for i, a in enumerate(state_order):
            for j, b in enumerate(state_order):
                rows.append(
                    {
                        "superwindow_id": row.get("superwindow_id", np.nan),
                        track_col: row.get(track_col, np.nan),
                        group_col: row.get(group_col, np.nan),
                        length_col: int(row[length_col]) if length_col in row.index and pd.notna(row[length_col]) else np.nan,
                        "from_state": a,
                        "to_state": b,
                        "transition_probability_per_superwindow": float(P[i, j]),
                        "n_transitions_in_superwindow": float(C[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_transitions_track_aware_across_lengths(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    track_col: str = "original_track_id",
    length_col: str = "superwindow_length",
    lengths: Sequence[int] | None = None,
    state_sequence_col: str = "state_sequence",
) -> pd.DataFrame:
    """
    Track-aware multi-length aggregation.

    1. Per (track, length): mean of per-superwindow P matrices
    2. Per (group, length): mean across tracks (each track one vote)
    3. Per group: mean across lengths (each length one vote)

    Avoids long tracks dominating via many short superwindows.
    """
    df = superwindows.to_pandas(superwindow_df)
    if lengths is not None:
        df = df.loc[df[length_col].astype(int).isin({int(L) for L in lengths})].copy()
    per_sw = _per_superwindow_transition_rows(
        df,
        group_col=group_col,
        state_order=state_order,
        state_sequence_col=state_sequence_col,
        track_col=track_col,
        length_col=length_col,
    )
    if per_sw.empty:
        return per_sw

    per_track = (
        per_sw.groupby([group_col, track_col, length_col, "from_state", "to_state"], dropna=False)[
            "transition_probability_per_superwindow"
        ]
        .mean()
        .reset_index(name="p_track_length")
    )
    per_group_length = (
        per_track.groupby([group_col, length_col, "from_state", "to_state"], dropna=False)
        .agg(
            transition_probability=("p_track_length", "mean"),
            n_tracks=(track_col, "nunique"),
        )
        .reset_index()
    )
    return aggregate_transitions_across_lengths(
        per_group_length.assign(n_superwindows=per_group_length["n_tracks"]),
        group_col=group_col,
        state_order=state_order,
        policy="equal_length_mean",
        length_col=length_col,
    ).assign(probability_mode="multi_length:track_equal_then_length_equal")


def composition_fractions_by_length(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    state_order: Sequence[Any],
    group_col: str,
    lengths: Sequence[int] | None = None,
    length_col: str = "superwindow_length",
    state_sequence_col: str = "state_sequence",
) -> pd.DataFrame:
    """State occupancy fractions computed separately at each length."""
    df = superwindows.to_pandas(superwindow_df)
    length_vals = sorted(int(x) for x in df[length_col].dropna().unique())
    if lengths is not None:
        want = {int(L) for L in lengths}
        length_vals = [L for L in length_vals if L in want]
    pieces = []
    for L in length_vals:
        sub = df.loc[df[length_col].astype(int) == L]
        if sub.empty:
            continue
        comp = superwindows.state_composition_fractions_from_sequences(
            sub,
            state_order=state_order,
            state_sequence_col=state_sequence_col,
            group_col=group_col,
        )
        comp = comp.copy()
        comp[length_col] = int(L)
        pieces.append(comp)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def aggregate_composition_across_lengths(
    per_length_comp_df: pd.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    policy: str = "equal_length_mean",
    length_col: str = "superwindow_length",
) -> pd.DataFrame:
    """Collapse per-length occupancy to one table for fan box sizing / flux."""
    df = per_length_comp_df.copy()
    if df.empty:
        return df
    weight_col = "n_windows" if policy != "equal_length_mean" and "n_windows" in df.columns else None
    rows = []
    for (g, state), sub in df.groupby([group_col, "state"], dropna=False):
        rows.append(
            {
                group_col: g,
                "state": state,
                "fraction": _weighted_mean_rows(sub, "fraction", weight_col),
                "n_windows": int(pd.to_numeric(sub.get("n_windows", 0), errors="coerce").fillna(0).sum()),
                "n_lengths": int(sub[length_col].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["state"] = pd.Categorical(out["state"], categories=list(state_order), ordered=True)
    out = out.sort_values([group_col, "state"]).reset_index(drop=True)
    out["state"] = out["state"].astype(str)
    return out


SW_UNIT_POLICIES = {
    "mean_per_superwindow",  # baseline: each SW equal (long tracks still contribute more SWs)
    "mean_per_track",  # SW→track→geno (equalize tracks)
    "mean_per_cell_of_tracks",  # SW→track→cell→geno
    "mean_per_file_of_tracks",  # SW→track→file→geno
    "cap_k_sw_per_track",  # keep ≤K SWs/track then mean_per_superwindow
    "balance_n_tracks",  # subsample tracks to min across groups, then mean_per_track
    "balance_n_cells",  # subsample cells to min across groups, then mean_per_cell_of_tracks
}


def _cap_superwindows_per_track(
    df: pd.DataFrame,
    *,
    track_col: str,
    max_sw_per_track: int,
    random_state: int = 0,
    length_col: str | None = "superwindow_length",
) -> pd.DataFrame:
    """Keep at most K superwindows per track (optionally within each length)."""
    if max_sw_per_track < 1:
        raise ValueError("max_sw_per_track must be >= 1")
    rng = np.random.default_rng(random_state)
    parts = []
    group_keys = [track_col]
    if length_col and length_col in df.columns:
        group_keys = [length_col, track_col]
    for _, sub in df.groupby(group_keys, sort=False):
        if len(sub) <= max_sw_per_track:
            parts.append(sub)
            continue
        idx = rng.choice(sub.index.to_numpy(), size=max_sw_per_track, replace=False)
        parts.append(sub.loc[idx])
    return pd.concat(parts, ignore_index=False).sort_index()


def _balance_units_across_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    unit_col: str,
    random_state: int = 0,
    length_col: str | None = "superwindow_length",
) -> pd.DataFrame:
    """Subsample units so each group has the same n_units (min across groups)."""
    rng = np.random.default_rng(random_state)
    parts = []
    length_vals = [None]
    if length_col and length_col in df.columns:
        length_vals = sorted(df[length_col].dropna().astype(int).unique())
    for L in length_vals:
        subL = df if L is None else df.loc[df[length_col].astype(int) == int(L)]
        unit_lists = {
            g: subL.loc[subL[group_col] == g, unit_col].dropna().unique().tolist()
            for g in subL[group_col].dropna().unique()
        }
        if not unit_lists:
            continue
        n_min = min(len(v) for v in unit_lists.values())
        if n_min < 1:
            continue
        keep_units = set()
        for g, units in unit_lists.items():
            chosen = rng.choice(np.array(units, dtype=object), size=n_min, replace=False)
            keep_units.update(chosen.tolist())
        parts.append(subL.loc[subL[unit_col].isin(keep_units)])
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=False)


def _per_sw_occupancy_rows(
    superwindow_df: pd.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    state_sequence_col: str = "state_sequence",
    track_col: str = "original_track_id",
    length_col: str = "superwindow_length",
    cell_col: str | None = None,
    file_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    meta_cols = [c for c in [cell_col, file_col] if c and c in superwindow_df.columns]
    for _, row in superwindow_df.iterrows():
        seq = [str(x) for x in list(row[state_sequence_col])]
        n = len(seq)
        counts = {str(s): 0 for s in state_order}
        for s in seq:
            if s in counts:
                counts[s] += 1
        base = {
            group_col: row[group_col],
            track_col: row.get(track_col, np.nan),
            "superwindow_id": row.get("superwindow_id", np.nan),
            length_col: int(row[length_col]) if length_col in row.index and pd.notna(row.get(length_col)) else np.nan,
        }
        for c in meta_cols:
            base[c] = row.get(c, np.nan)
        for s in state_order:
            rows.append(
                {
                    **base,
                    "state": s,
                    "n_windows": counts[str(s)],
                    "fraction": (counts[str(s)] / n) if n else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _collapse_sw_unit_means(
    per_sw_value_df: pd.DataFrame,
    *,
    group_col: str,
    track_col: str,
    value_col: str,
    key_cols: Sequence[str],
    unit_policy: str,
    cell_col: str | None,
    file_col: str | None,
) -> pd.DataFrame:
    """
    Collapse per-superwindow values to group-level means under unit_policy.

    ``key_cols`` are the matrix keys besides group (e.g. from_state,to_state or state).
    """
    keys = list(key_cols)

    if unit_policy in {"mean_per_superwindow", "cap_k_sw_per_track"}:
        out = (
            per_sw_value_df.groupby([group_col] + keys, dropna=False)[value_col]
            .mean()
            .reset_index(name="value")
        )
        nmap = per_sw_value_df.groupby(group_col)["superwindow_id"].nunique()
        out["n_units"] = out[group_col].map(nmap)
        return out

    # SW → track
    per_track = (
        per_sw_value_df.groupby([group_col, track_col] + keys, dropna=False)[value_col]
        .mean()
        .reset_index(name="p_track")
    )

    if unit_policy in {"mean_per_track", "balance_n_tracks"}:
        out = (
            per_track.groupby([group_col] + keys, dropna=False)["p_track"].mean().reset_index(name="value")
        )
        nmap = per_track.groupby(group_col)[track_col].nunique()
        out["n_units"] = out[group_col].map(nmap)
        return out

    if unit_policy in {"mean_per_cell_of_tracks", "balance_n_cells"}:
        if not cell_col or cell_col not in per_sw_value_df.columns:
            raise KeyError(f"unit_policy={unit_policy!r} requires cell_col")
        # attach cell onto per_track via track meta
        track_meta = (
            per_sw_value_df[[group_col, track_col, cell_col]]
            .drop_duplicates([group_col, track_col])
        )
        per_track = per_track.merge(track_meta, on=[group_col, track_col], how="left")
        per_cell = (
            per_track.groupby([group_col, cell_col] + keys, dropna=False)["p_track"]
            .mean()
            .reset_index(name="p_cell")
        )
        out = per_cell.groupby([group_col] + keys, dropna=False)["p_cell"].mean().reset_index(name="value")
        nmap = per_cell.groupby(group_col)[cell_col].nunique()
        out["n_units"] = out[group_col].map(nmap)
        return out

    if unit_policy == "mean_per_file_of_tracks":
        if not file_col or file_col not in per_sw_value_df.columns:
            raise KeyError("unit_policy='mean_per_file_of_tracks' requires file_col")
        track_meta = (
            per_sw_value_df[[group_col, track_col, file_col]]
            .drop_duplicates([group_col, track_col])
        )
        per_track = per_track.merge(track_meta, on=[group_col, track_col], how="left")
        per_file = (
            per_track.groupby([group_col, file_col] + keys, dropna=False)["p_track"]
            .mean()
            .reset_index(name="p_file")
        )
        out = per_file.groupby([group_col] + keys, dropna=False)["p_file"].mean().reset_index(name="value")
        nmap = per_file.groupby(group_col)[file_col].nunique()
        out["n_units"] = out[group_col].map(nmap)
        return out

    raise ValueError(f"Unknown unit_policy={unit_policy!r}")


def estimate_superwindow_transitions_unit_weighted(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    unit_policy: str = "mean_per_cell_of_tracks",
    track_col: str = "original_track_id",
    cell_col: str | None = "cell",
    file_col: str | None = "filename",
    length_col: str = "superwindow_length",
    state_sequence_col: str = "state_sequence",
    lengths: Sequence[int] | None = None,
    max_sw_per_track: int = 1,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    Estimate transitions from superwindows with explicit unit weighting / balancing.

    Applied **within each length**, returning a per-length probability table plus
    a note that length amalgamation is separate (``build_multi_length_switch_tables``).
    """
    if unit_policy not in SW_UNIT_POLICIES:
        raise ValueError(f"Unknown unit_policy={unit_policy!r}; choose from {sorted(SW_UNIT_POLICIES)}")

    df = superwindows.to_pandas(superwindow_df)
    if cell_col and cell_col not in df.columns:
        cell_col = None
    if file_col and file_col not in df.columns:
        file_col = None
    if track_col not in df.columns:
        raise KeyError(f"track_col={track_col!r} missing from superwindow dataframe")
    if length_col not in df.columns:
        df = df.copy()
        df[length_col] = int(df["n_windows"].iloc[0]) if "n_windows" in df.columns else 1

    if lengths is not None:
        df = df.loc[df[length_col].astype(int).isin({int(L) for L in lengths})].copy()

    # Optional balancing / capping (modifies which SWs enter)
    work = df
    if unit_policy == "cap_k_sw_per_track":
        work = _cap_superwindows_per_track(
            work,
            track_col=track_col,
            max_sw_per_track=max_sw_per_track,
            random_state=random_state,
            length_col=length_col,
        )
    elif unit_policy == "balance_n_tracks":
        work = _balance_units_across_groups(
            work,
            group_col=group_col,
            unit_col=track_col,
            random_state=random_state,
            length_col=length_col,
        )
    elif unit_policy == "balance_n_cells":
        if cell_col is None:
            raise KeyError("balance_n_cells requires a cell column")
        work = _balance_units_across_groups(
            work,
            group_col=group_col,
            unit_col=cell_col,
            random_state=random_state,
            length_col=length_col,
        )

    # effective collapse policy after balancing/capping
    collapse_policy = {
        "cap_k_sw_per_track": "mean_per_superwindow",
        "balance_n_tracks": "mean_per_track",
        "balance_n_cells": "mean_per_cell_of_tracks",
    }.get(unit_policy, unit_policy)

    per_sw = _per_superwindow_transition_rows(
        work,
        group_col=group_col,
        state_order=state_order,
        state_sequence_col=state_sequence_col,
        track_col=track_col,
        length_col=length_col,
    )
    # attach cell/file onto per_sw for hierarchy
    meta_cols = [c for c in [cell_col, file_col] if c]
    if meta_cols:
        meta = work[["superwindow_id", track_col, group_col] + meta_cols].drop_duplicates("superwindow_id")
        per_sw = per_sw.merge(meta, on=["superwindow_id", track_col, group_col], how="left")

    per_occ = _per_sw_occupancy_rows(
        work,
        group_col=group_col,
        state_order=state_order,
        state_sequence_col=state_sequence_col,
        track_col=track_col,
        length_col=length_col,
        cell_col=cell_col,
        file_col=file_col,
    )

    prob_parts = []
    comp_parts = []
    for L, sub_p in per_sw.groupby(length_col, dropna=False):
        collapsed = _collapse_sw_unit_means(
            sub_p,
            group_col=group_col,
            track_col=track_col,
            value_col="transition_probability_per_superwindow",
            key_cols=["from_state", "to_state"],
            unit_policy=collapse_policy,
            cell_col=cell_col,
            file_col=file_col,
        )
        collapsed = collapsed.rename(columns={"value": "transition_probability"})
        collapsed[length_col] = int(L) if pd.notna(L) else L
        collapsed["transition_pair"] = (
            collapsed["from_state"].astype(str) + " -> " + collapsed["to_state"].astype(str)
        )
        collapsed["is_self_transition"] = (
            collapsed["from_state"].astype(str) == collapsed["to_state"].astype(str)
        )
        collapsed["probability_mode"] = f"sw_unit:{unit_policy}"
        collapsed["n_superwindows"] = collapsed.get("n_units", np.nan)
        prob_parts.append(collapsed)

        sub_o = per_occ.loc[per_occ[length_col] == L] if pd.notna(L) else per_occ
        # for occupancy collapse, reuse same hierarchy on fraction
        occ_c = _collapse_sw_unit_means(
            sub_o,
            group_col=group_col,
            track_col=track_col,
            value_col="fraction",
            key_cols=["state"],
            unit_policy=collapse_policy,
            cell_col=cell_col,
            file_col=file_col,
        ).rename(columns={"value": "fraction"})
        occ_c[length_col] = int(L) if pd.notna(L) else L
        n_win = sub_o.groupby([group_col, "state"], dropna=False)["n_windows"].sum().reset_index()
        occ_c = occ_c.merge(n_win, on=[group_col, "state"], how="left")
        comp_parts.append(occ_c)

    per_length_prob = pd.concat(prob_parts, ignore_index=True) if prob_parts else pd.DataFrame()
    per_length_comp = pd.concat(comp_parts, ignore_index=True) if comp_parts else pd.DataFrame()
    return {
        "per_length_prob_df": per_length_prob,
        "per_length_comp_df": per_length_comp,
        "used_superwindow_df": work,
        "unit_policy": unit_policy,
        "collapse_policy": collapse_policy,
        "max_sw_per_track": max_sw_per_track,
        "n_superwindows_used": int(len(work)),
        "n_superwindows_input": int(len(df)),
    }


def build_multi_length_switch_tables(
    superwindow_df: pd.DataFrame | pl.DataFrame,
    *,
    group_col: str,
    state_order: Sequence[Any],
    policy: str = "equal_length_mean",
    mode: str = "mean_per_superwindow",
    lengths: Sequence[int] | None = None,
    length_col: str = "superwindow_length",
    track_col: str = "original_track_id",
    unit_policy: str | None = None,
    cell_col: str | None = "cell",
    file_col: str | None = "filename",
    max_sw_per_track: int = 1,
    random_state: int = 0,
) -> dict[str, Any]:
    """
    One-stop helper: per-length tables + aggregated switch/composition tables.

    Downstream fan/heatmap code can use ``switch_prob_df`` and ``comp_df`` exactly
    as in the fixed-length notebook.

    Parameters
    ----------
    unit_policy
        Optional superwindow **unit weighting** applied within each length before
        length amalgamation. See ``SW_UNIT_POLICIES``.
        If None, uses legacy behaviour via ``mode`` / ``track_equal_then_length_equal``.
    """
    if policy not in LENGTH_AGG_POLICIES:
        raise ValueError(f"Unknown policy={policy!r}; choose from {sorted(LENGTH_AGG_POLICIES)}")

    unit_info = None
    if unit_policy is not None:
        unit_info = estimate_superwindow_transitions_unit_weighted(
            superwindow_df,
            group_col=group_col,
            state_order=state_order,
            unit_policy=unit_policy,
            track_col=track_col,
            cell_col=cell_col,
            file_col=file_col,
            length_col=length_col,
            lengths=lengths,
            max_sw_per_track=max_sw_per_track,
            random_state=random_state,
        )
        per_length_prob = unit_info["per_length_prob_df"]
        per_length_comp = unit_info["per_length_comp_df"]
        # length amalgamation on unit-weighted per-length tables
        if policy == "track_equal_then_length_equal":
            # already track-aware within length if unit_policy is track/cell; just equal-mean lengths
            switch_prob_df = aggregate_transitions_across_lengths(
                per_length_prob,
                group_col=group_col,
                state_order=state_order,
                policy="equal_length_mean",
                lengths=lengths,
                length_col=length_col,
            )
            switch_prob_df["probability_mode"] = f"multi_length:{unit_policy}+equal_length_mean"
            comp_policy = "equal_length_mean"
        else:
            switch_prob_df = aggregate_transitions_across_lengths(
                per_length_prob,
                group_col=group_col,
                state_order=state_order,
                policy=policy,
                lengths=lengths,
                length_col=length_col,
            )
            switch_prob_df["probability_mode"] = f"multi_length:{unit_policy}+{policy}"
            comp_policy = policy if policy in {"equal_length_mean", "n_superwindow_weighted", "transition_count_weighted"} else "equal_length_mean"
    else:
        per_length_prob = transition_probabilities_by_length(
            superwindow_df,
            group_col=group_col,
            state_order=state_order,
            mode=mode,
            lengths=lengths,
            length_col=length_col,
        )
        per_length_comp = composition_fractions_by_length(
            superwindow_df,
            state_order=state_order,
            group_col=group_col,
            lengths=lengths,
            length_col=length_col,
        )

        if policy == "track_equal_then_length_equal":
            switch_prob_df = aggregate_transitions_track_aware_across_lengths(
                superwindow_df,
                group_col=group_col,
                state_order=state_order,
                track_col=track_col,
                length_col=length_col,
                lengths=lengths,
            )
            comp_policy = "equal_length_mean"
        else:
            switch_prob_df = aggregate_transitions_across_lengths(
                per_length_prob,
                group_col=group_col,
                state_order=state_order,
                policy=policy,
                lengths=lengths,
                length_col=length_col,
            )
            comp_policy = policy

    comp_df = aggregate_composition_across_lengths(
        per_length_comp,
        group_col=group_col,
        state_order=state_order,
        policy=comp_policy if comp_policy in {"equal_length_mean", "n_superwindow_weighted", "transition_count_weighted"} else "equal_length_mean",
        length_col=length_col,
    )
    return {
        "per_length_prob_df": per_length_prob,
        "per_length_comp_df": per_length_comp,
        "switch_prob_df": switch_prob_df,
        "comp_df": comp_df,
        "policy": policy,
        "mode": mode,
        "unit_policy": unit_policy,
        "unit_info": unit_info,
    }
