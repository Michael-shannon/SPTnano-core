"""General plots: gate heatmaps (plot_xy_heatmap), stacked bars, histograms, boxplots and track galleries."""

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
import re
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LogNorm, is_color_like
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from . import config

def _set_svg_fonts():
    # Keep SVG text editable in Illustrator; Helvetica if installed, else Arial.
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = "sans-serif"
    if any("Helvetica" in f.name for f in fm.fontManager.ttflist):
        mpl.rcParams["font.sans-serif"] = ["Helvetica"]
    else:
        mpl.rcParams["font.sans-serif"] = ["Arial"]


_set_svg_fonts()


def use_paper_style(font_size=14):
    """Matplotlib state the paper figures were drawn in: whitegrid, seaborn notebook context (6 pt ticks), Arial."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size})
    _set_svg_fonts()


def center_scale_data(
    df,
    columns,
    method='standard',
    group_by=None,
    copy=True
):
    """
    Center and scale data using various scaling methods from scikit-learn.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to scale.
    columns : list of str
        List of column names to scale.
    method : str, optional
        Scaling method to use. Options are:
        - 'standard': StandardScaler (mean=0, std=1) [default]
        - 'minmax': MinMaxScaler (scales to [0, 1])
        - 'robust': RobustScaler (uses median and IQR, robust to outliers)
        - 'maxabs': MaxAbsScaler (scales to [-1, 1], preserves sparsity)
        - 'quantile': QuantileTransformer (uniform or gaussian distribution)
        - 'power': PowerTransformer (makes data more Gaussian)
        - 'normalize': Normalizer (scales each sample to unit norm)
        - 'standard_minmax': StandardScaler then MinMaxScaler (center then bound to [0,1])
    group_by : str or list of str, optional
        Column name(s) to group by before scaling. If provided, scaling is done
        independently for each group. Default is None.
    copy : bool, optional
        Whether to return a copy of the dataframe. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with scaled columns. Original columns are replaced with scaled values.
    
    Examples
    --------
    >>> # Standard scaling
    >>> df_scaled = center_scale_data(df, ['x_um', 'y_um'], method='standard')
    
    >>> # MinMax scaling by group
    >>> df_scaled = center_scale_data(df, ['speed'], method='minmax', group_by='condition')
    
    >>> # Robust scaling (good for data with outliers)
    >>> df_scaled = center_scale_data(df, features_list, method='robust')
    
    """
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        MaxAbsScaler,
        QuantileTransformer,
        PowerTransformer,
        Normalizer
    )
    
    if copy:
        df = df.copy()
    
    # Initialize the scaler based on method
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler(),
        'maxabs': MaxAbsScaler(),
        'quantile': QuantileTransformer(),
        'power': PowerTransformer(),
        'normalize': Normalizer(),
        'standard_minmax': 'combined'  # Special case
    }
    
    if method not in scalers:
        raise ValueError(
            f"method must be one of {list(scalers.keys())}, got '{method}'"
        )
    
    # Ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    # Check if all columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")
    
    # Scale data
    if group_by is None:
        # Scale all data at once
        if method == 'standard_minmax':
            # First apply StandardScaler, then MinMaxScaler
            scaler1 = StandardScaler()
            df[columns] = scaler1.fit_transform(df[columns])
            scaler2 = MinMaxScaler()
            df[columns] = scaler2.fit_transform(df[columns])
        else:
            scaler = scalers[method]
            df[columns] = scaler.fit_transform(df[columns])
    else:
        # Scale by group
        if isinstance(group_by, str):
            group_by = [group_by]
        
        for group_vals, group_df in df.groupby(group_by):
            if method == 'standard_minmax':
                # First apply StandardScaler, then MinMaxScaler
                scaler1 = StandardScaler()
                df.loc[group_df.index, columns] = scaler1.fit_transform(group_df[columns])
                scaler2 = MinMaxScaler()
                df.loc[group_df.index, columns] = scaler2.fit_transform(df.loc[group_df.index, columns])
            else:
                scaler = scalers[method]
                df.loc[group_df.index, columns] = scaler.fit_transform(group_df[columns])
    
    return df


def _get_location_column(df):
    """
    Helper function to detect whether dataframe has 'Location' or 'location' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for location column

    Returns
    -------
    str
        The correct column name ('Location' or 'location')

    Raises
    ------
    ValueError
        If neither 'Location' nor 'location' column is found

    """
    if "Location" in df.columns:
        return "Location"
    elif "location" in df.columns:
        return "location"
    else:
        raise ValueError(
            "Neither 'Location' nor 'location' column found in dataframe. Available columns: "
            + str(list(df.columns))
        )


def plot_stacked_bar_svg(
    df,
    x_category,
    stack_category=None,
    order=None,
    stack_order=None,
    title=None,
    font_size=16,
    colormap="Dark2",
    figsize=(10, 8),
    background="white",
    transparent=False,
    line_color="black",
    export_format="png",
    master_dir=None,
    custom="_",
    show_plot=True,
    return_svg=False,
    show_percentages=True,
    percentage_threshold=5.0,
    ylabel="Percentage (%)",
    legend_title=None,
    grid=False,
):
    """
    Plot a generalized stacked bar chart showing the percentage distribution of any categorical variable.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing the data with the specified categorical columns.
    x_category : str
        The column name of the category to plot on the x-axis.
    stack_category : str, optional
        The column name of the category to stack. If None, defaults to 'motion_class' for backward compatibility.
    order : list, optional
        Custom order for the categories on the x-axis. Default is None.
    stack_order : list, optional
        Custom order for the stacked categories. Default is None.
    title : str, optional
        Title for the plot. If None, generates a default title.
    font_size : int, optional
        Font size for the plot text. Default is 16.
    colormap : str, optional
        Colormap for the plot. Default is 'Dark2'. (If 'colorblind', uses seaborn's colorblind palette.)
        You may also pass an explicit list/tuple of colors (e.g. hex strings) to use for the
        stacked categories, in the same order as ``stack_order`` (if provided) or the discovered
        categories.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).
    background : str, optional
        Background color for the figure. Default is 'white'.
    transparent : bool, optional
        If True, makes the plot background transparent. Default is False.
    line_color : str, optional
        Color of all lines (axis lines, bar outlines, gridlines, and text). Default is 'black'.
    export_format : str, optional
        File format to export the figure ('png' or 'svg'). Default is 'png'.
    master_dir : str, optional
        Directory to save the plot. If not provided, defaults to "plots".
    custom : str, optional
        Custom string to add to filename. Default is "_".
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the post-processed SVG image data as a string.
    show_percentages : bool, optional
        Whether to annotate percentages on bars. Default is True.
    percentage_threshold : float, optional
        Only show percentage labels on bars with height >= this threshold. Default is 5.0.
    ylabel : str, optional
        Label for y-axis. Default is "Percentage (%)".
    legend_title : str, optional
        Title for the legend. If None, uses the stack_category name.
    grid : bool, optional
        Whether to show grid lines. Default is False.

    Returns
    -------
    str or None
        When export_format is 'svg' and return_svg is True, returns the cleaned SVG data as a string.
        Otherwise, returns None.

    """
    # Apply background and transparency settings.
    if transparent:
        figure_background = "none"
        axis_background = (0, 0, 0, 0)  # Transparent background for axes
    else:
        figure_background = background
        axis_background = background

    # Set default stack_category for backward compatibility
    if stack_category is None:
        stack_category = "motion_class"
        
    # Check if required columns exist
    if x_category not in df.columns:
        raise ValueError(f"Column '{x_category}' not found in dataframe")
    if stack_category not in df.columns:
        raise ValueError(f"Column '{stack_category}' not found in dataframe")

    # Apply custom order if provided
    if order is not None:
        df[x_category] = pd.Categorical(df[x_category], categories=order, ordered=True)
    
    if stack_order is not None:
        df[stack_category] = pd.Categorical(df[stack_category], categories=stack_order, ordered=True)

    # Calculate percentage data for each stack category
    percentage_data = (
        df.groupby([x_category, stack_category])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum() * 100, axis=1)
    )

    # Determine unique stack categories and assign colors
    stack_categories = list(percentage_data.columns) if stack_order is None else list(stack_order)
    percentage_data = percentage_data.reindex(columns=stack_categories, fill_value=0)
    if isinstance(colormap, dict):
        missing_colors = [cat for cat in stack_categories if cat not in colormap]
        if missing_colors:
            raise ValueError(
                f"colormap dictionary is missing colors for stack categories: {missing_colors}"
            )
        colors = [colormap[cat] for cat in stack_categories]
    elif isinstance(colormap, (list, tuple, np.ndarray)):
        colors = list(colormap)
        if len(colors) < len(stack_categories):
            raise ValueError(
                f"colormap list has {len(colors)} colors but needs at least {len(stack_categories)} "
                f"for stack_category={stack_category!r}."
            )
        colors = colors[: len(stack_categories)]
    elif colormap == "colorblind":
        colors = sns.color_palette("colorblind", len(stack_categories))
    elif colormap == "Dark2":
        cmap = cm.get_cmap("Dark2")
        base_colors = list(cmap.colors) if hasattr(cmap, "colors") else list(cmap(np.linspace(0, 1, 8)))
        if len(base_colors) < len(stack_categories):
            repeats = int(np.ceil(len(stack_categories) / len(base_colors)))
            base_colors = base_colors * repeats
        colors = base_colors[: len(stack_categories)]
    else:
        colors = plt.get_cmap(colormap, len(stack_categories)).colors

    # Create figure and axes.
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    ax.set_facecolor(axis_background)

    # Plot the stacked bar chart.
    percentage_data.plot(
        kind="bar", stacked=True, ax=ax, color=colors, edgecolor=line_color
    )

    # Add black outlines to each bar.
    for patch in ax.patches:
        patch.set_edgecolor(line_color)

    # Annotate percentages on the bars if enabled
    if show_percentages:
        for patch in ax.patches:
            width, height = patch.get_width(), patch.get_height()
            x, y = patch.get_xy()
            if height >= percentage_threshold:  # Only annotate if above threshold
                ax.annotate(
                    f"{height:.1f}%",
                    (x + width / 2, y + height / 2),
                    ha="center",
                    va="center",
                    fontsize=max(8, font_size - 2),  # Slightly smaller font for percentages
                    color=line_color,
                    weight="bold",
                )

    # Set title (generate default if not provided)
    if title is None:
        title = f"Distribution of {stack_category.replace('_', ' ').title()}"
    
    # Customize text elements
    ax.set_title(title, fontsize=font_size, color=line_color, pad=20)
    ax.set_xlabel(x_category.replace('_', ' ').title(), fontsize=font_size, color=line_color)
    ax.set_ylabel(ylabel, fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
    )

    # Rotate x-tick labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Set legend title
    if legend_title is None:
        legend_title = stack_category.replace('_', ' ').title()

    # Move the legend outside the plot
    legend = plt.legend(
        title=legend_title,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title_fontsize=font_size,
        prop={"size": font_size},
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(line_color)
    legend.get_title().set_color(line_color)

    # Add grid if requested
    if grid:
        ax.grid(
            True, which="both", linestyle="--", linewidth=0.5, color=line_color, zorder=0
        )

    # Customize axis spines.
    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        if transparent:
            spine.set_alpha(0.5)

    plt.tight_layout()

    # Saving/export functionality.
    svg_data = None
    master_dir = master_dir or "plots"
    os.makedirs(master_dir, exist_ok=True)
    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    filename = f"{master_dir}/stacked_bar_{x_category}_{custom}.{ext}"
    plt.savefig(filename, bbox_inches="tight", transparent=transparent, format=ext)

    if ext == "svg":
        with open(filename, encoding="utf-8") as f:
            svg_data = f.read()
        # Remove any <clipPath> definitions and clip-path attributes.
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
        # Remove the <metadata> section.
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
        # Remove XML declaration and DOCTYPE.
        svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
        svg_data = svg_data.strip()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


def plot_tracks_grid_by_cluster(
    tracks_df,
    cluster_col="cluster_id",
    segments_per_cluster=5,
    window_size=config.OVERLAP,  # Number of frames per window segment
    filename=None,
    file_id=None,
    location=None,
    condition=None,
    time_start=None,
    time_end=None,
    motion_type=None,
    master_dir=config.MASTER,
    transparent_background=True,
    save_path=None,
    pixel_size_um=config.PIXELSIZE_MICRONS,
    frame_interval=config.TIME_BETWEEN_FRAMES,
    colorway="tab20",
    order=None,  # Order for cluster coloring, defaults to low-to-high cluster numbers
    figsize=(12, 8),
    track_size_um=5,  # Size of each track's bounding box in microns
    line_thickness=0.6,
    dpi=200,
    export_format="svg",
    return_svg=False,
    show_plot=True,
    grid_spacing_um=2,  # Spacing between columns in microns
    track_spacing_um=1,  # Spacing between tracks within a column
    random_seed=42,
    show_cluster_labels=True,
    label_fontsize=12,
):
    """
    Create a grid visualization of track window segments organized by cluster ID.

    Each column represents a different cluster, with multiple window segments from that cluster
    displayed in a vertical arrangement. Each segment shows a fixed number of frames (window_size)
    to ensure consistent cluster ID throughout the segment. Segments are centered and normalized
    rather than showing absolute spatial positions.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with cluster assignments
    cluster_col : str, default 'cluster_id'
        Column name containing cluster IDs
    segments_per_cluster : int, default 5
        Number of window segments to display per cluster
    window_size : int, default 30
        Number of consecutive frames per window segment (ensures consistent cluster ID)
    filename, file_id, location, condition : str, optional
        Filters for selecting specific data
    time_start, time_end : float, optional
        Time range to display (in seconds)
    motion_type : str, optional
        Filter by motion type if specified
    track_size_um : float, default 5
        Size of each track segment's display area in microns
    grid_spacing_um : float, default 2
        Horizontal spacing between cluster columns in microns
    track_spacing_um : float, default 1
        Vertical spacing between segments within a column in microns
    random_seed : int, default 42
        Seed for reproducible segment selection
    show_cluster_labels : bool, default True
        Whether to show cluster labels at the top of each column
    label_fontsize : int, default 12
        Font size for cluster labels
    order : list, optional
        Custom order for cluster coloring. If None, uses low-to-high cluster numbers
    ... (other parameters same as plot_tracks_static_svg)

    Returns
    -------
    dict : Dictionary containing segment information and layout details

    """
    # Set random seed for reproducible results
    np.random.seed(random_seed)

    # Apply same filtering logic as plot_tracks_static_svg
    df = tracks_df.copy()

    # Filter by location
    if location is not None:
        location_col = _get_location_column(df)
        if location not in df[location_col].unique():
            available_locations = df[location_col].unique()
            raise ValueError(
                f"Location '{location}' not found. Available: {available_locations}"
            )
        df = df[df[location_col] == location]

    # Filter by condition
    if condition is not None:
        if condition not in df["condition"].unique():
            available_conditions = df["condition"].unique()
            raise ValueError(
                f"Condition '{condition}' not found. Available: {available_conditions}"
            )
        df = df[df["condition"] == condition]

    # Filter by filename
    if filename is not None:
        if filename not in df["filename"].unique():
            available_files = df["filename"].unique()
            raise ValueError(
                f"Filename '{filename}' not found. Available: {available_files}"
            )
        df = df[df["filename"] == filename]

    # Filter by file_id
    if file_id is not None:
        if "file_id" in df.columns:
            df = df[df["file_id"] == file_id]

    # Filter by motion type
    if motion_type is not None:
        if "motion_class" in df.columns:
            df = df[df["motion_class"] == motion_type]

    # Time filtering
    if time_start is not None or time_end is not None:
        if "time_s" not in df.columns:
            df["time_s"] = df["frame"] * frame_interval

        if time_start is not None:
            df = df[df["time_s"] >= time_start]
        if time_end is not None:
            df = df[df["time_s"] <= time_end]

    # Check if cluster column exists
    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column '{cluster_col}' not found in dataframe. Available columns: {list(df.columns)}"
        )

    # Get unique clusters, excluding NaN values (use same logic as other functions)
    all_unique_clusters = sorted([c for c in df[cluster_col].unique() if pd.notna(c)])
    n_clusters = len(all_unique_clusters)

    if n_clusters == 0:
        raise ValueError(
            "No valid clusters found after filtering (all values are NaN or missing)"
        )

    # Set up cluster ordering for left-to-right positioning (consistent with other functions)
    if order is None:
        ordered_clusters = all_unique_clusters  # Low to high cluster numbers
    else:
        # Use provided order, but only include clusters that exist in data
        ordered_clusters = [c for c in order if c in all_unique_clusters]
        # Add any remaining clusters not in the provided order
        remaining_clusters = [
            c for c in all_unique_clusters if c not in ordered_clusters
        ]
        ordered_clusters.extend(remaining_clusters)

    # Find valid window segments for each cluster
    selected_segments = {}
    segment_info = []

    for cluster_id in all_unique_clusters:
        cluster_df = df[(df[cluster_col] == cluster_id) & pd.notna(df[cluster_col])]

        # Find valid window segments within this cluster
        segment_candidates = []
        for track_id in cluster_df["unique_id"].unique():
            track_data = cluster_df[cluster_df["unique_id"] == track_id].sort_values(
                "frame"
            )
            frames = track_data["frame"].values

            # Find consecutive frame runs of at least window_size length
            if len(frames) < window_size:
                continue

            # Check for consecutive frames
            frame_diffs = np.diff(frames)
            split_points = np.where(frame_diffs != 1)[0] + 1
            frame_runs = np.split(frames, split_points)

            for run in frame_runs:
                if len(run) >= window_size:
                    # Find all possible window starts in this run
                    for start_idx in range(len(run) - window_size + 1):
                        start_frame = run[start_idx]
                        end_frame = run[start_idx + window_size - 1]

                        # Verify all frames in window have the same cluster ID
                        window_data = track_data[
                            (track_data["frame"] >= start_frame)
                            & (track_data["frame"] <= end_frame)
                        ]
                        if (
                            len(window_data) == window_size
                            and len(window_data[cluster_col].unique()) == 1
                        ):
                            segment_candidates.append(
                                {
                                    "track_id": track_id,
                                    "start_frame": start_frame,
                                    "end_frame": end_frame,
                                    "cluster_id": cluster_id,
                                }
                            )

        # Sample segments for this cluster
        n_available = len(segment_candidates)
        if n_available == 0:
            selected_segments[cluster_id] = []
            continue

        if n_available >= segments_per_cluster:
            sampled_indices = np.random.choice(
                n_available, segments_per_cluster, replace=False
            )
        else:
            # Use all available and pad with repeats if needed
            sampled_indices = list(range(n_available))
            while len(sampled_indices) < segments_per_cluster:
                additional_needed = min(
                    segments_per_cluster - len(sampled_indices), n_available
                )
                sampled_indices.extend(
                    np.random.choice(n_available, additional_needed, replace=False)
                )

        selected_segments[cluster_id] = [
            segment_candidates[i] for i in sampled_indices[:segments_per_cluster]
        ]

        # Store segment info
        for i, segment in enumerate(selected_segments[cluster_id]):
            segment_info.append(
                {
                    "cluster_id": cluster_id,
                    "track_id": segment["track_id"],
                    "segment_index": i,
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "n_frames": window_size,
                }
            )

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Calculate grid layout using ordered clusters
    max_segments = (
        max(len(segments) for segments in selected_segments.values())
        if selected_segments
        else 0
    )
    if max_segments == 0:
        raise ValueError("No valid segments found for any cluster")

    # Use ordered clusters for width calculation
    n_ordered_clusters = len(ordered_clusters)
    total_width = (
        n_ordered_clusters * track_size_um + (n_ordered_clusters - 1) * grid_spacing_um
    )
    total_height = max_segments * track_size_um + (max_segments - 1) * track_spacing_um

    # Set up cluster-based coloring (same logic as plot_tracks_static_svg)

    cluster_to_color = {}
    for i, cluster_id in enumerate(ordered_clusters):
        cluster_to_color[cluster_id] = plt.cm.get_cmap(colorway)(
            i / max(len(ordered_clusters) - 1, 1)
        )

    # FIRST PASS: Calculate global coordinate range for consistent scaling
    all_coord_ranges = []
    all_segments_data = []

    for cluster_id in ordered_clusters:
        if (
            cluster_id not in selected_segments
            or len(selected_segments[cluster_id]) == 0
        ):
            continue

        for segment in selected_segments[cluster_id]:
            # Get segment data (window frames only)
            segment_data = df[
                (df["unique_id"] == segment["track_id"])
                & (df["frame"] >= segment["start_frame"])
                & (df["frame"] <= segment["end_frame"])
            ].sort_values("frame")

            if len(segment_data) == 0:
                continue

            # Convert to microns and center
            x_coords = segment_data["x"].values * pixel_size_um
            y_coords = segment_data["y"].values * pixel_size_um

            # Center the segment
            x_coords = x_coords - np.mean(x_coords)
            y_coords = y_coords - np.mean(y_coords)

            # Store data for second pass
            all_segments_data.append(
                {
                    "cluster_id": cluster_id,
                    "segment": segment,
                    "x_coords": x_coords,
                    "y_coords": y_coords,
                }
            )

            # Calculate range for this segment
            coord_range = max(np.ptp(x_coords), np.ptp(y_coords))
            if coord_range > 0:
                all_coord_ranges.append(coord_range)

    # Calculate global scale factor
    if all_coord_ranges:
        global_max_range = max(all_coord_ranges)
        global_scale_factor = (
            track_size_um * 0.8
        ) / global_max_range  # Use 80% of available space
    else:
        global_scale_factor = 1.0

    # SECOND PASS: Plot segments using global scale
    for segment_data in all_segments_data:
        cluster_id = segment_data["cluster_id"]
        segment = segment_data["segment"]
        x_coords = segment_data["x_coords"].copy()
        y_coords = segment_data["y_coords"].copy()

        # Find this segment's position in the grid
        cluster_idx = ordered_clusters.index(cluster_id)
        segment_idx = selected_segments[cluster_id].index(segment)

        cluster_x_center = (
            cluster_idx * (track_size_um + grid_spacing_um) + track_size_um / 2
        )
        segment_y_center = (
            segment_idx * (track_size_um + track_spacing_um) + track_size_um / 2
        )
        cluster_color = cluster_to_color[cluster_id]

        # Apply global scale factor
        x_coords = x_coords * global_scale_factor
        y_coords = y_coords * global_scale_factor

        # Position in grid
        x_coords = x_coords + cluster_x_center
        y_coords = y_coords + segment_y_center

        # Plot segment with cluster color
        ax.plot(
            x_coords, y_coords, color=cluster_color, linewidth=line_thickness, alpha=0.8
        )

        # Add a small dot at the start
        ax.plot(
            x_coords[0],
            y_coords[0],
            "o",
            color=cluster_color,
            markersize=line_thickness * 2,
            alpha=0.6,
        )

    # Add cluster labels using ordered clusters
    if show_cluster_labels:
        for cluster_idx, cluster_id in enumerate(ordered_clusters):
            cluster_x_center = (
                cluster_idx * (track_size_um + grid_spacing_um) + track_size_um / 2
            )
            label_y = total_height + 0.5
            ax.text(
                cluster_x_center,
                label_y,
                f"Cluster {cluster_id}",
                ha="center",
                va="bottom",
                fontsize=label_fontsize,
                fontweight="bold",
            )

    # Set axis properties
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-0.5, total_height + (1 if show_cluster_labels else 0.5))
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    title_parts = []
    if condition:
        title_parts.append(f"Condition: {condition}")
    if location:
        title_parts.append(f"Location: {location}")
    if filename:
        title_parts.append(f"File: {filename}")
    if time_start is not None or time_end is not None:
        time_str = f"Time: {time_start or 0:.1f}-{time_end or 'end':.1f}s"
        title_parts.append(time_str)

    if title_parts:
        ax.set_title(" | ".join(title_parts), pad=20, fontsize=label_fontsize)

    # Set background
    if transparent_background:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    # Save or show
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create filename
        save_filename = f"segment_grid_{cluster_col}"
        if condition:
            save_filename += f"_{condition}"
        if location:
            save_filename += f"_{location}"
        if filename:
            save_filename += f"_{filename}"
        save_filename += f".{export_format}"

        full_path = os.path.join(save_path, save_filename)

        if export_format.lower() == "svg":
            plt.savefig(
                full_path,
                format="svg",
                bbox_inches="tight",
                transparent=transparent_background,
                dpi=dpi,
            )
            if return_svg:
                with open(full_path) as f:
                    svg_content = f.read()
        else:
            plt.savefig(
                full_path,
                format=export_format,
                bbox_inches="tight",
                transparent=transparent_background,
                dpi=dpi,
            )

    if show_plot:
        plt.show()
    else:
        plt.close()

    # Return information
    result = {
        "segment_info": segment_info,
        "clusters": all_unique_clusters,
        "ordered_clusters": ordered_clusters,  # The actual left-to-right order used
        "selected_segments": selected_segments,
        "n_clusters": n_clusters,
        "segments_per_cluster": segments_per_cluster,
        "window_size": window_size,
        "grid_dimensions": (total_width, total_height),
    }

    if save_path and export_format.lower() == "svg" and return_svg:
        result["svg_content"] = svg_content

    return result


def plot_xy_heatmap(
    df,
    x_col,
    y_col,
    plot_type='hexbin',
    color_by=None,
    order=None,
    small_multiples=False,
    shared_scale=True,
    gridsize=50,
    figsize=(8, 6),
    cmap=None,
    alpha=0.6,
    s=10,
    title=None,
    xlabel=None,
    ylabel=None,
    scale_data=False,
    scale_method='standard',
    scaler=None,
    scale_group_by=None,
    xlim=None,
    ylim=None,
    save_path=None,
    dpi=150,
    log_scale=True,
    transparent_background=False,
    line_color='black',
    export_format='png',
    return_svg=False,
    contour_levels=None,
    contour_cmap='viridis',
    gates=None,
    gate_order=None,
    gate_colors='black',
    gate_linewidth=2,
    gate_alpha=0.1,
    gate_edge_alpha=1.0,
    gate_text_size=10,
    gate_label_border=True,
    gate_linestyle='--',
    gate_fill_color=None,
    gate_label_position='auto',
    **kwargs
):
    """
    Create flexible x vs y plots with hexbin or scatter options.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame containing the data to plot. Accepts both pandas and polars.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    plot_type : str, optional
        Type of plot. Options are:
        - 'hexbin': Hexagonal binning (good for medium-large datasets)
        - 'scatter': Individual points (good for small datasets or categorical data)
        Default is 'hexbin'.
    color_by : str, optional
        Column name to color/separate data by. If None, pools all data together.
        If specified, colors scatter plots by category or creates small multiples
        when small_multiples=True. Default is None.
    order : list, optional
        Custom order for categories when using color_by parameter. If None, 
        categories will be sorted alphabetically. If provided, should be a list 
        of category values in the desired display order. Only categories present 
        in both the order list and the data will be plotted. Default is None.
    small_multiples : bool, optional
        If True and color_by is provided, creates separate subplots for each category.
        If False, plots all data together (colored by category for scatter).
        Default is False.
    shared_scale : bool, optional
        If True (default), all subplots share the same x/y scale (max of all data).
        If False, each subplot uses different colormaps (for hexbin/hist2d).
        Only applies when small_multiples=True.
    gridsize : int, optional
        Grid size for hexbin plots. Default is 50.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
    cmap : str or list of str, optional
        Colormap(s) for plots. 
        - For hexbin plots:
          * Single hexbin: string colormap name (default: 'inferno')
          * Small multiples hexbin: list of colormaps, one per subplot
            (default: ['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'hot'])
        - For scatter plots:
          * Continuous color_by (numeric, >20 unique values): colormap name 
            (default: 'viridis'), displays with colorbar
          * Categorical color_by: converts colormap to discrete colors, or pass 
            list of colors directly
          * Special value 'colorblind': Wong 2011 colorblind-friendly palette
          * No color_by: ignored (uses default color)
        Good colormaps: 'colorblind' (recommended!), 'viridis', 'plasma', 
        'inferno', 'magma', 'cividis', 'coolwarm', 'RdYlBu', 'tab10', 'Set2'.
    alpha : float, optional
        Transparency for scatter plots. Default is 0.6.
    s : float or int, optional
        Marker size for scatter plots. Default is 10.
    title : str, optional
        Overall figure title. Default is None.
    xlabel : str, optional
        X-axis label. Default is x_col.
    ylabel : str, optional
        Y-axis label. Default is y_col.
    scale_data : bool, optional
        If True, applies scaling to x_col and y_col before plotting. Default is False.
    scale_method : str, optional
        Scaling method to use if scale_data=True and scaler is None. Default is 'standard'.
        Options: 'standard', 'minmax', 'robust', 'standard_minmax', etc.
    scaler : sklearn scaler object, optional
        Pre-fitted scaler to use for transforming data. If provided, this scaler
        will be used instead of fitting a new one (CRITICAL for Option A gating).
        The scaler must have a .transform() method and be already fitted.
        Default is None.
    scale_group_by : str or list of str, optional
        Group by column(s) for scaling. Only used if scaler is None. Default is None.
    xlim : tuple, optional
        X-axis limits (xmin, xmax). Default is None (auto).
    ylim : tuple, optional
        Y-axis limits (ymin, ymax). Default is None (auto).
    save_path : str, optional
        Path to save the figure. Can be:
        - A directory path: filename will be auto-generated as 
          "{x_col}_vs_{y_col}_{plot_type}_by_{color_by}_{modifiers}.{ext}"
        - A full file path: will be used as-is (extension adjusted to match export_format)
        Default is None (no save).
    dpi : int, optional
        DPI for saved figure. Default is 150.
    log_scale : bool, optional
        If True, use logarithmic color scaling for hexbin plots to better
        highlight areas of low and high density. Default is True.
    transparent_background : bool, optional
        If True, makes the background transparent. Useful for presentations
        and publications. Default is False.
    line_color : str, optional
        Color for axis lines, ticks, and labels. Default is 'black'.
        Use 'white' for dark backgrounds in presentations.
    export_format : str, optional
        File format to export ('png' or 'svg'). Default is 'png'.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the cleaned SVG data as a string.
        Default is False.
    contour_levels : int or None, optional
        If specified, adds density contour lines on top of scatter/hexbin plots.
        Number indicates how many contour levels to draw. Default is None (no contours).
    contour_cmap : str, optional
        Colormap for contour lines when contour_levels is specified.
        Default is 'viridis'.
    gates : ROIManager or list, optional
        Gates to draw on the plot. Can be:
        - An ROIManager object (e.g., from interactive_roi_gating_with_capture)
        - A list of gate dictionaries with 'coordinates', 'type', and optional 'name'
        Default is None (no gates drawn).
    gate_order : list of str, optional
        Order of gate names for color assignment. If provided, gates will be colored
        in this order. Only gates present in both gate_order and the data will be drawn.
        This ensures consistent color mapping across multiple plots. Default is None
        (uses order from ROIManager or alphabetical by gate name).
    gate_colors : str or list of str, optional
        Color(s) for gate edges/borders. Can be a single color or list of colors.
        Good options for dashed lines: 'black', 'darkgray', 'navy', 'darkgreen',
        'darkred', 'purple'. Default is 'black'.
    gate_linewidth : float, optional
        Line width for gate borders. Default is 2.
    gate_alpha : float, optional
        Transparency for gate fill (0-1). Set to 0 for no fill. Default is 0.1.
        This does NOT affect edge transparency - edges use gate_edge_alpha.
    gate_edge_alpha : float, optional
        Transparency for gate edges/borders (0-1). Default is 1.0 (fully opaque).
        Keep at 1.0 for clear, visible gate boundaries.
    gate_text_size : float, optional
        Font size for gate labels. Default is 10.
    gate_label_border : bool, optional
        Whether to draw a border around gate label text boxes. Default is True.
    gate_linestyle : str, optional
        Line style for gate borders. Options: '-' (solid), '--' (dashed), 
        '-.' (dash-dot), ':' (dotted). Default is '--' (dashed).
    gate_fill_color : str or None, optional
        Fill color for gates. If None, uses same color as edge with alpha.
        Set to 'none' for transparent fill. Good options: 'lightgray', 'lightblue',
        'lightyellow', 'white', or 'none'. Default is None (uses edge color).
    gate_label_position : str, optional
        Label position strategy. Options:
        - 'auto': Smart positioning based on gate location (recommended)
          * Gates closer to left side (0) → label at top-left corner
          * Gates closer to right side (1) → label at top-right corner
          * Always positions at top of gate for visibility
        - 'top-left': Always top-left corner
        - 'top-right': Always top-right corner
        - 'center': Center of gate
        Default is 'auto'.
    **kwargs
        Additional keyword arguments passed to plotting functions.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : matplotlib.axes.Axes or np.ndarray of Axes
        The axes object(s).
    svg_data : str (optional)
        If export_format='svg' and return_svg=True, returns cleaned SVG string.
    
    Examples
    --------
    >>> # Simple hexbin plot (all data pooled)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um')
    
    >>> # Scatter plot colored by category (automatic colors)
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', color_by='mol')
    
    >>> # Scatter plot with colorblind-friendly palette (recommended!)
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', 
    ...                            color_by='mol', cmap='colorblind')
    
    >>> # Scatter plot with other custom colormap for categories
    >>> fig, ax = plot_xy_heatmap(df, 'speed', 'D', plot_type='scatter', 
    ...                            color_by='mol', cmap='Set2')
    
    >>> # Scatter plot with continuous colormap (for numeric color_by)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um', plot_type='scatter',
    ...                            color_by='speed', cmap='plasma')
    
    >>> # Small multiples by condition
    >>> fig, axes = plot_xy_heatmap(df, 'x_um', 'y_um', 
    ...                              color_by='condition',
    ...                              small_multiples=True)
    
    >>> # Small multiples with custom order
    >>> fig, axes = plot_xy_heatmap(df, 'x_um', 'y_um', 
    ...                              color_by='condition',
    ...                              order=['control', 'drug1', 'drug2'],
    ...                              small_multiples=True)
    
    >>> # With data scaling
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            scale_data=True,
    ...                            scale_method='standard_minmax')
    
    >>> # With gates (dashed lines, no fill, opaque edges)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='black',
    ...                            gate_linestyle='--',
    ...                            gate_fill_color='none',
    ...                            gate_linewidth=2,
    ...                            gate_edge_alpha=1.0)  # Opaque edges (default)
    
    >>> # With gates (transparent fill, smart auto label positioning, separate alphas)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='navy',
    ...                            gate_fill_color='lightblue',
    ...                            gate_alpha=0.15,          # Fill transparency
    ...                            gate_edge_alpha=1.0,      # Opaque edges
    ...                            gate_label_position='auto')  # Smart: left gates→top-left, right gates→top-right
    
    >>> # With gates (no label borders for clean look)
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_colors='black',
    ...                            gate_linestyle='--',
    ...                            gate_fill_color='none',
    ...                            gate_label_border=False)  # No borders around labels
    
    >>> # With gates (consistent color mapping across plots)
    >>> gate_names = ['Low_Speed', 'Medium_Speed', 'High_Speed', 'Very_High']
    >>> gate_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']
    >>> fig, ax = plot_xy_heatmap(df, 'x_um', 'y_um',
    ...                            gates=roi_manager,
    ...                            gate_order=gate_names,  # Consistent order
    ...                            gate_colors=gate_colors,  # Matches order
    ...                            gate_linestyle='--')
    
    """
    # Detect and convert Polars DataFrames to Pandas for plotting
    # Matplotlib requires pandas/numpy for plotting operations
    try:
        import polars as pl
        is_polars_input = isinstance(df, pl.DataFrame)
    except ImportError:
        is_polars_input = False
    
    if is_polars_input:
        # Convert to pandas for plotting (matplotlib requires pandas/numpy)
        plot_df = df.to_pandas()
    else:
        # Create a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()
    
    # Apply scaling if requested
    if scale_data:
        if scaler is not None:
            # Use the provided scaler (OPTION A - for population-level gating)
            plot_df[[x_col, y_col]] = scaler.transform(plot_df[[x_col, y_col]])
        else:
            # Fit a new scaler on this data (default behavior)
            plot_df = center_scale_data(
                plot_df,
                columns=[x_col, y_col],
                method=scale_method,
                group_by=scale_group_by,
                copy=False
            )
    
    # Set default labels
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
    
    def add_density_contours(ax, x, y, levels=5, cmap_name='viridis', alpha_contour=0.6):
        """Add 2D density contours."""
        # Calculate 2D density
        try:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            
            # Create grid for contours
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Add padding
            x_min -= x_range * 0.1
            x_max += x_range * 0.1
            y_min -= y_range * 0.1
            y_max += y_range * 0.1
            
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(gaussian_kde(xy)(positions).T, xx.shape)
            
            # Draw contours
            contours = ax.contour(xx, yy, f, levels=levels, cmap=cmap_name, 
                                 alpha=alpha_contour, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=8)
        except (ValueError, np.linalg.LinAlgError):
            # Skip if KDE fails (too few points or singular matrix)
            pass
    
    # Set default colormaps
    default_hexbin_cmaps = ['inferno', 'viridis', 'plasma', 'magma', 'cividis', 'hot']
    if cmap is None:
        if plot_type == 'hexbin':
            cmap = 'inferno'  # Single plot default
        else:
            cmap = None  # Scatter uses categorical colors
    
    # Set backgrounds
    figure_background = 'none' if transparent_background else 'white'
    axis_background = (0, 0, 0, 0) if transparent_background else 'white'
    
    # Determine if we need subplots based on color_by and small_multiples
    if color_by is not None and small_multiples:
        if order is not None:
            # Use custom order, filtering to only include categories present in data
            unique_vals = set(plot_df[color_by].unique())
            categories = [cat for cat in order if cat in unique_vals]
        else:
            # Default: sort alphabetically
            categories = sorted(plot_df[color_by].unique())
        n_cats = len(categories)
        
        # Always create vertical stack of plots for small multiples
        fig, axes = plt.subplots(
            n_cats, 1,
            figsize=(figsize[0], figsize[1] * n_cats)
        )
        fig.patch.set_facecolor(figure_background)
        if n_cats == 1:
            axes = [axes]
        
        # Calculate shared scale if needed
        if shared_scale:
            x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
            y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()
            # Use max range for both axes
            data_range = max(x_max - x_min, y_max - y_min)
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            xlim_shared = (x_center - data_range/2, x_center + data_range/2)
            ylim_shared = (y_center - data_range/2, y_center + data_range/2)
        
        # Handle colormap list for hexbin plots
        cmap_list = cmap
        if plot_type == 'hexbin':
            if isinstance(cmap, str):
                # Single colormap provided - use different ones for each subplot
                cmap_list = default_hexbin_cmaps
            if isinstance(cmap_list, list):
                # Extend list if needed
                if len(cmap_list) < n_cats:
                    cmap_list = (cmap_list * (n_cats // len(cmap_list) + 1))[:n_cats]
            else:
                cmap_list = [cmap_list] * n_cats
        
        # Plot each category
        for idx, cat in enumerate(categories):
            ax = axes[idx]
            ax.set_facecolor(axis_background)
            cat_df = plot_df[plot_df[color_by] == cat]
            
            if plot_type == 'hexbin':
                # Use different colormap for each subplot
                current_cmap = cmap_list[idx] if isinstance(cmap_list, list) else cmap_list
                norm = LogNorm() if log_scale else None
                
                hb = ax.hexbin(
                    cat_df[x_col],
                    cat_df[y_col],
                    gridsize=gridsize,
                    cmap=current_cmap,
                    mincnt=1,
                    norm=norm,
                    **kwargs
                )
                cbar = plt.colorbar(hb, ax=ax, label='Count (log scale)' if log_scale else 'Count')
                cbar.ax.yaxis.set_tick_params(color=line_color)
                cbar.ax.yaxis.label.set_color(line_color)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, cat_df[x_col].values, cat_df[y_col].values, 
                                        levels=contour_levels, cmap_name=contour_cmap)
                
            elif plot_type == 'scatter':
                # For scatter in small multiples, use uniform single color per subplot
                # Get color from specified palette or use default
                if cmap == 'colorblind':
                    # Use Wong 2011 colorblind-friendly palette
                    colorblind_colors = [
                        '#0173B2',  # Blue
                        '#DE8F05',  # Orange  
                        '#029E73',  # Green
                        '#CC78BC',  # Purple
                        '#CA9161',  # Brown
                        '#FBAFE4',  # Pink
                        '#949494',  # Gray
                        '#ECE133',  # Yellow
                        '#56B4E9',  # Sky blue
                        '#D55E00'   # Vermillion
                    ]
                    color_palette = colorblind_colors[idx % len(colorblind_colors)]
                elif cmap is not None and isinstance(cmap, (list, tuple)):
                    # Custom list of colors
                    color_palette = cmap[idx % len(cmap)]
                elif cmap is not None and isinstance(cmap, str):
                    # Try to use as colormap
                    try:
                        colormap = plt.cm.get_cmap(cmap)
                        color_palette = colormap(idx / n_cats)
                    except:
                        color_palette = plt.cm.tab10(idx % 10)
                else:
                    # Default to tab10
                    color_palette = plt.cm.tab10(idx % 10)
                
                ax.scatter(cat_df[x_col], cat_df[y_col], s=s, alpha=alpha, 
                          color=color_palette, **kwargs)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, cat_df[x_col].values, cat_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
            else:
                raise ValueError(f"plot_type must be 'hexbin' or 'scatter', got '{plot_type}'")
            
            # Set limits
            if shared_scale:
                ax.set_xlim(xlim_shared if xlim is None else xlim)
                ax.set_ylim(ylim_shared if ylim is None else ylim)
            else:
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
            
            # Labels and styling
            ax.set_xlabel(xlabel, color=line_color)
            ax.set_ylabel(ylabel, color=line_color)
            ax.set_title(f"{color_by}: {cat}", color=line_color)
            ax.grid(alpha=0.3, color=line_color)
            ax.tick_params(colors=line_color)
            for spine in ax.spines.values():
                spine.set_edgecolor(line_color)
        
        # No unused axes to hide in vertical layout
        
    else:
        # Single plot (pool all data or color by category)
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(figure_background)
        ax.set_facecolor(axis_background)
        
        if plot_type == 'hexbin':
            # Density plot - pool all data
            norm = LogNorm() if log_scale else None
            hb = ax.hexbin(
                plot_df[x_col],
                plot_df[y_col],
                gridsize=gridsize,
                cmap=cmap,
                mincnt=1,
                norm=norm,
                **kwargs
            )
            cbar = plt.colorbar(hb, ax=ax, label='Count (log scale)' if log_scale else 'Count')
            cbar.ax.yaxis.set_tick_params(color=line_color)
            cbar.ax.yaxis.label.set_color(line_color)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
            
            # Add contours if requested
            if contour_levels is not None:
                add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                    levels=contour_levels, cmap_name=contour_cmap)
            
        elif plot_type == 'scatter':
            if color_by is not None and color_by in plot_df.columns:
                # Check if color_by is numeric (continuous) or categorical
                is_numeric_color = np.issubdtype(plot_df[color_by].dtype, np.number)
                n_unique = plot_df[color_by].nunique()
                
                # Treat as continuous if numeric and many unique values
                if is_numeric_color and n_unique > 20:
                    # Continuous coloring with colormap
                    scatter_cmap = cmap if cmap is not None else 'viridis'
                    scatter = ax.scatter(
                        plot_df[x_col],
                        plot_df[y_col],
                        c=plot_df[color_by],
                        s=s,
                        alpha=alpha,
                        cmap=scatter_cmap,
                        **kwargs
                    )
                    cbar = plt.colorbar(scatter, ax=ax, label=color_by)
                    cbar.ax.yaxis.set_tick_params(color=line_color)
                    cbar.ax.yaxis.label.set_color(line_color)
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=line_color)
                else:
                    # Categorical coloring
                    if order is not None:
                        # Use custom order, filtering to only include categories present in data
                        unique_vals = set(plot_df[color_by].unique())
                        unique_colors = [cat for cat in order if cat in unique_vals]
                    else:
                        # Default: sort alphabetically
                        unique_colors = sorted(plot_df[color_by].unique())
                    
                    # Get colors from colormap if provided, otherwise use default palette
                    if cmap is not None:
                        # Handle special palette names
                        if cmap == 'colorblind':
                            # Use Wong 2011 colorblind-friendly palette
                            colors = [
                                '#0173B2',  # Blue
                                '#DE8F05',  # Orange  
                                '#029E73',  # Green
                                '#CC78BC',  # Purple
                                '#CA9161',  # Brown
                                '#FBAFE4',  # Pink
                                '#949494',  # Gray
                                '#ECE133',  # Yellow
                                '#56B4E9',  # Sky blue
                                '#D55E00'   # Vermillion
                            ]
                        elif isinstance(cmap, str):
                            # Try to get it as a matplotlib colormap
                            try:
                                colormap = plt.cm.get_cmap(cmap)
                                colors = [colormap(i / len(unique_colors)) for i in range(len(unique_colors))]
                            except ValueError:
                                # If not a valid colormap, try as seaborn palette
                                try:
                                    import seaborn as sns
                                    colors = sns.color_palette(cmap, n_colors=len(unique_colors))
                                except:
                                    # Fall back to default
                                    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                        else:
                            colors = cmap  # Assume it's a list of colors
                    else:
                        # Use matplotlib's default color cycle
                        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    
                    for idx, color_val in enumerate(unique_colors):
                        color_df = plot_df[plot_df[color_by] == color_val]
                        ax.scatter(
                            color_df[x_col],
                            color_df[y_col],
                            s=s,
                            alpha=alpha,
                            color=colors[idx % len(colors)],
                            label=str(color_val),
                            **kwargs
                        )
                    ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add contours if requested (on all data)
                if contour_levels is not None:
                    add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
            else:
                # Single color scatter (pool all data)
                # For single scatter without color_by, cmap is not used
                ax.scatter(plot_df[x_col], plot_df[y_col], s=s, alpha=alpha, **kwargs)
                
                # Add contours if requested
                if contour_levels is not None:
                    add_density_contours(ax, plot_df[x_col].values, plot_df[y_col].values,
                                        levels=contour_levels, cmap_name=contour_cmap)
        else:
            raise ValueError(f"plot_type must be 'hexbin' or 'scatter', got '{plot_type}'")
        
        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Labels and styling
        ax.set_xlabel(xlabel, color=line_color)
        ax.set_ylabel(ylabel, color=line_color)
        ax.grid(alpha=0.3, color=line_color)
        ax.tick_params(colors=line_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
        axes = ax
    
    # Draw gates if provided
    if gates is not None:
        # Handle both ROIManager object and list of gate dictionaries
        gate_list = []
        if hasattr(gates, 'rois'):  # ROIManager object
            # If scale_data is False and ROIManager has a scaler, we need to inverse transform
            # the gate coordinates to match the unscaled plot
            if not scale_data and hasattr(gates, 'scaler') and gates.scaler is not None:
                gate_list = gates.get_inverse_transformed_gates()
            else:
                gate_list = gates.rois
        elif isinstance(gates, list):  # List of gate dicts
            gate_list = gates
        else:
            print(f"Warning: gates parameter type not recognized. Expected ROIManager or list.")
        
        # Apply gate_order if provided
        if gate_order is not None:
            # Create mapping of gate names to gates
            gate_dict = {gate.get('name', f"Gate {i}"): gate for i, gate in enumerate(gate_list)}
            # Reorder gates according to gate_order, filtering to only include present gates
            ordered_gates = []
            for gate_name in gate_order:
                if gate_name in gate_dict:
                    ordered_gates.append(gate_dict[gate_name])
            gate_list = ordered_gates
        
        # Draw gates on all axes
        axes_to_draw = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
        
        from matplotlib.patches import Rectangle
        from matplotlib.patches import Polygon as MplPolygon
        import matplotlib.colors as mcolors
        
        for ax_obj in axes_to_draw:
            # Get axis limits for smart positioning
            x_lim = ax_obj.get_xlim()
            y_lim = ax_obj.get_ylim()
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            x_mid = (x_lim[0] + x_lim[1]) / 2
            y_mid = (y_lim[0] + y_lim[1]) / 2
            
            for i, gate in enumerate(gate_list):
                coords = gate.get('coordinates', [])
                gate_name = gate.get('name', f"Gate {i}")
                
                if not coords:
                    continue
                
                # Determine gate edge color
                if isinstance(gate_colors, str):
                    edge_color_base = gate_colors
                elif isinstance(gate_colors, list):
                    edge_color_base = gate_colors[i % len(gate_colors)]
                else:
                    edge_color_base = 'black'
                
                # Convert edge color to RGBA with edge_alpha baked in
                edge_rgba = mcolors.to_rgba(edge_color_base, alpha=gate_edge_alpha)
                
                # Determine gate fill color with fill_alpha baked in
                if gate_fill_color == 'none':
                    fill_rgba = 'none'
                elif gate_fill_color is not None:
                    # Convert fill color to RGBA with gate_alpha baked in
                    fill_rgba = mcolors.to_rgba(gate_fill_color, alpha=gate_alpha)
                else:
                    # Use edge color with gate_alpha for fill
                    fill_rgba = mcolors.to_rgba(edge_color_base, alpha=gate_alpha)
                
                # Draw based on gate type
                if gate.get('type') == 'rect':
                    # Rectangle gate
                    x0, y0 = coords[0]
                    x1, y1 = coords[2]  # Opposite corner
                    width = x1 - x0
                    height = y1 - y0
                    
                    rect = Rectangle(
                        (x0, y0), width, height,
                        linewidth=gate_linewidth,
                        edgecolor=edge_rgba,  # RGBA with edge_alpha baked in
                        facecolor=fill_rgba,  # RGBA with fill_alpha baked in (or 'none')
                        linestyle=gate_linestyle,
                        label=gate_name,
                        clip_on=True,
                    )
                    # Don't set alpha parameter - it would multiply with the baked-in alphas!
                    ax_obj.add_patch(rect)
                    
                    # Smart label positioning
                    gate_center_x = x0 + width / 2
                    gate_center_y = y0 + height / 2
                    
                    if gate_label_position == 'auto':
                        # Smart positioning based on gate proximity to 0 and 1 on axes
                        # Calculate how close gate is to min (0-like) vs max (1-like) on each axis
                        
                        # For X-axis: determine if gate is closer to left (0) or right (1)
                        dist_to_left = gate_center_x - x_lim[0]
                        dist_to_right = x_lim[1] - gate_center_x
                        
                        # For Y-axis: we always want labels at TOP of gates for visibility
                        # but adjust horizontal position based on X proximity
                        
                        if dist_to_left < dist_to_right:
                            # Gate closer to left/0 side → label at top-left
                            label_x = x0 + width * 0.05
                            ha = 'left'
                        else:
                            # Gate closer to right/1 side → label at top-right
                            label_x = x1 - width * 0.05
                            ha = 'right'
                        
                        # Always put labels at top of gate for better visibility
                        label_y = y1 - height * 0.05
                        va = 'top'
                    elif gate_label_position == 'top-left':
                        label_x = x0 + width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'left'
                        va = 'top'
                    elif gate_label_position == 'top-right':
                        label_x = x1 - width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'right'
                        va = 'top'
                    elif gate_label_position == 'center':
                        label_x = gate_center_x
                        label_y = gate_center_y
                        ha = 'center'
                        va = 'center'
                    else:
                        # Default to top-left
                        label_x = x0 + width * 0.05
                        label_y = y1 - height * 0.05
                        ha = 'left'
                        va = 'top'
                    
                    # Add label
                    if gate_label_border:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor=edge_color_base, linewidth=1)
                    else:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor='none', linewidth=0)
                    
                    ax_obj.text(
                        label_x,
                        label_y,
                        gate_name,
                        fontsize=gate_text_size,
                        color=edge_color_base,  # Use base color for text
                        fontweight='bold',
                        ha=ha,
                        va=va,
                        bbox=label_bbox
                    )
                else:
                    # Polygon gate
                    polygon = MplPolygon(
                        coords,
                        linewidth=gate_linewidth,
                        edgecolor=edge_rgba,  # RGBA with edge_alpha baked in
                        facecolor=fill_rgba,  # RGBA with fill_alpha baked in (or 'none')
                        linestyle=gate_linestyle,
                        label=gate_name,
                        clip_on=True,
                    )
                    # Don't set alpha parameter - it would multiply with the baked-in alphas!
                    ax_obj.add_patch(polygon)
                    
                    # Add label for polygon
                    xs, ys = zip(*coords)
                    cx, cy = np.mean(xs), np.mean(ys)
                    
                    # Smart positioning for polygon
                    if gate_label_position == 'auto':
                        # Use same logic as rectangles: proximity to left vs right
                        dist_to_left = cx - x_lim[0]
                        dist_to_right = x_lim[1] - cx
                        
                        if dist_to_left < dist_to_right:
                            # Polygon closer to left/0 side → position label to left
                            ha = 'right'
                            label_offset_x = -0.02 * x_range
                        else:
                            # Polygon closer to right/1 side → position label to right
                            ha = 'left'
                            label_offset_x = 0.02 * x_range
                        
                        # Position slightly above centroid
                        va = 'bottom'
                        label_offset_y = 0.02 * y_range
                        
                        label_x = cx + label_offset_x
                        label_y = cy + label_offset_y
                    else:
                        label_x = cx
                        label_y = cy
                        ha = 'center'
                        va = 'center'
                    
                    if gate_label_border:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor=edge_color_base, linewidth=1)
                    else:
                        label_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                         edgecolor='none', linewidth=0)
                    
                    ax_obj.text(
                        label_x,
                        label_y,
                        gate_name,
                        fontsize=gate_text_size,
                        color=edge_color_base,  # Use base color for text
                        fontweight='bold',
                        ha=ha,
                        va=va,
                        bbox=label_bbox
                    )
    
    # Set overall title
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold', color=line_color)
    
    plt.tight_layout()
    
    # Save if requested
    svg_data = None
    if save_path is not None:
        ext = export_format.lower()
        if ext not in ['png', 'svg']:
            print("Invalid export format specified. Defaulting to 'png'.")
            ext = 'png'
        
        # Check if save_path is a directory or a full file path
        if os.path.isdir(save_path) or (not os.path.exists(save_path) and not save_path.endswith(('.png', '.svg'))):
            # It's a directory - generate filename automatically
            save_dir = save_path
            
            # Build descriptive filename
            filename_parts = [f"{x_col}_vs_{y_col}", plot_type]
            if color_by is not None:
                filename_parts.append(f"by_{color_by}")
            if small_multiples:
                filename_parts.append("multiples")
            if scale_data:
                filename_parts.append(scale_method)
            
            filename = "_".join(filename_parts) + f".{ext}"
            save_path = os.path.join(save_dir, filename)
        else:
            # It's a full file path - use as is but ensure correct extension
            save_dir = os.path.dirname(save_path)
            if not save_path.endswith(f'.{ext}'):
                save_path = save_path.rsplit('.', 1)[0] + f'.{ext}' if '.' in save_path else f'{save_path}.{ext}'
        
        # Create directory if it doesn't exist
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig.savefig(
            save_path,
            transparent=transparent_background,
            dpi=dpi,
            format=ext,
            bbox_inches='tight'
        )
        
        # SVG post-processing
        if ext == 'svg':
            try:
                with open(save_path, encoding='utf-8') as f:
                    svg_data = f.read()
                # Remove <clipPath> definitions, metadata, XML declaration, and DOCTYPE
                svg_data = re.sub(
                    r'<clipPath id="[^"]*">.*?</clipPath>', '', svg_data, flags=re.DOTALL
                )
                svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', '', svg_data)
                svg_data = re.sub(
                    r'<metadata>.*?</metadata>', '', svg_data, flags=re.DOTALL
                )
                svg_data = re.sub(r'<\?xml[^>]*\?>', '', svg_data, flags=re.DOTALL)
                svg_data = re.sub(r'<!DOCTYPE[^>]*>', '', svg_data, flags=re.DOTALL)
                svg_data = svg_data.strip()
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(svg_data)
            except Exception as e:
                print(f"Warning: SVG post-processing failed: {e}")
        
        print(f"Figure saved to: {save_path}")
    
    if return_svg and svg_data:
        return fig, axes, svg_data
    else:
        return fig, axes


def _process_palette(palette, n_colors):
    """
    Process palette parameter that can be either a string (palette name) or a list of colors.
    
    Parameters
    ----------
    palette : str or list
        Either a palette name ('colorblind', 'Dark2', etc.) or a list of colors (hex codes, named colors, RGB tuples)
    n_colors : int
        Number of colors needed
        
    Returns
    -------
    list
        List of colors, cycling if needed
    """
    if isinstance(palette, (list, tuple)):
        # Custom colors provided - cycle if needed
        return [palette[i % len(palette)] for i in range(n_colors)]
    elif palette == "colorblind":
        # Use Wong 2011 colorblind-friendly palette
        colorblind_colors = [
            '#0173B2',  # Blue
            '#DE8F05',  # Orange  
            '#029E73',  # Green
            '#CC78BC',  # Purple
            '#CA9161',  # Brown
            '#FBAFE4',  # Pink
            '#949494',  # Gray
            '#ECE133',  # Yellow
            '#56B4E9',  # Sky blue
            '#D55E00'   # Vermillion
        ]
        return [colorblind_colors[i % len(colorblind_colors)] for i in range(n_colors)]
    else:
        # Use seaborn/matplotlib palette
        import seaborn as sns
        return sns.color_palette(palette, n_colors)


def _is_polars(df):
    """Check if dataframe is a Polars DataFrame."""
    return isinstance(df, pl.DataFrame)


def _get_unique_categories(df, column, order=None):
    """Get unique categories from a column, handling both Polars and Pandas."""
    if _is_polars(df):
        if order is not None:
            return order
        return df[column].unique().to_list()
    else:
        # Pandas logic with categorical
        if not isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype("category")
        if order is not None:
            df[column] = pd.Categorical(df[column], categories=order, ordered=True)
        return df[column].cat.categories.tolist()


def _filter_dataframe(df, column, value):
    """Filter dataframe by column value, handling both Polars and Pandas."""
    if _is_polars(df):
        return df.filter(pl.col(column) == value)
    else:
        return df[df[column] == value]


def _filter_range(df, column, lower, upper):
    """Filter dataframe by range, handling both Polars and Pandas."""
    if _is_polars(df):
        return df.filter((pl.col(column) >= lower) & (pl.col(column) <= upper))
    else:
        return df[(df[column] >= lower) & (df[column] <= upper)]


def _column_min(df, column):
    """Get column minimum, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].min()
    else:
        return df[column].min()


def _column_max(df, column):
    """Get column maximum, handling both Polars and Pandas."""
    if _is_polars(df):
        return df[column].max()
    else:
        return df[column].max()


def _create_log_column(df, feature, new_feature, log_base):
    """Create a log-transformed column, handling both Polars and Pandas."""
    if _is_polars(df):
        if log_base == 10:
            return df.with_columns(pl.col(feature).log10().alias(new_feature))
        elif log_base == 2:
            return df.with_columns(pl.col(feature).log().alias(new_feature) / np.log(2))
        else:
            return df.with_columns(pl.col(feature).log().alias(new_feature) / np.log(log_base))
    else:
        # Pandas - modify in place
        if log_base == 10:
            df[new_feature] = np.log10(df[feature])
        elif log_base == 2:
            df[new_feature] = np.log2(df[feature])
        else:
            df[new_feature] = np.log(df[feature]) / np.log(log_base)
        return df


def _to_numpy(series):
    """Convert a series to numpy array, handling both Polars and Pandas."""
    if isinstance(series, pl.Series):
        return series.to_numpy()
    elif isinstance(series, pl.DataFrame):
        # If single column dataframe
        return series.to_numpy().flatten()
    else:
        # Pandas
        return series.values


def _groupby_agg(df, groupby_col, agg_col, agg_func):
    """Perform groupby aggregation, handling both Polars and Pandas."""
    if _is_polars(df):
        if agg_func == 'median':
            result = df.group_by(groupby_col).agg(pl.col(agg_col).median())
        elif agg_func == 'mean':
            result = df.group_by(groupby_col).agg(pl.col(agg_col).mean())
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")
        # Return as dict for consistency
        return dict(zip(result[groupby_col].to_list(), result[agg_col].to_list()))
    else:
        if agg_func == 'median':
            return df.groupby(groupby_col)[agg_col].median()
        elif agg_func == 'mean':
            return df.groupby(groupby_col)[agg_col].mean()
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")


def plot_histograms(
    data_df,
    feature,
    bins=100,
    separate=None,
    xlimit=None,
    small_multiples=False,
    palette="colorblind",  # Can be string (palette name) or list of colors
    use_kde=False,
    kde_fill=True,
    show_plot=True,
    master_dir=None,
    tick_interval=5,
    average="mean",
    order=None,
    grid=False,
    background="white",
    transparent=False,
    condition_colors=None,
    line_color="black",
    font_size=9,
    showavg=True,
    export_format="png",
    return_svg=False,
    x_range=None,
    y_range=None,
    percentage=True,
    log_scale=False,
    log_base=10,
    alpha=1,
    log_axis_label="log",
    save_folder=None,
    figsize=(3, 3),
):
    """
    Plot histograms with flexible color options and customization.
    
    Compatible with both Polars and Pandas DataFrames.
    
    Parameters
    ----------
    data_df : DataFrame (Polars or Pandas)
        Input dataframe containing the data to plot.
    palette : str or list, default "colorblind"
        Color palette for categories. Can be:
        - String: palette name ('colorblind' uses Wong 2011, 'Dark2', 'Set2', etc.)
        - List: custom colors as hex codes, named colors, or RGB tuples
        Example: ['#0173B2', '#DE8F05', '#029E73'] or ['red', 'blue', 'green']
    
    (Other parameters documented elsewhere)
    """
    # Detect dataframe type
    is_polars_df = _is_polars(data_df)
    
    if master_dir is None:
        master_dir = "plots"

    baseline_width = 3.0
    scale_factor = figsize[0] / baseline_width
    scaled_font = font_size * scale_factor
    plt.rcParams.update(
        {
            "font.size": scaled_font,
            "axes.titlesize": scaled_font,
            "axes.labelsize": scaled_font,
            "xtick.labelsize": scaled_font,
            "ytick.labelsize": scaled_font,
        }
    )

    if log_scale:
        new_feature = "log_" + feature
        # Check for non-positive values
        if is_polars_df:
            has_nonpositive = (data_df[feature] <= 0).any()
            if has_nonpositive:
                num_negative = (data_df[feature] <= 0).sum()
                min_val = data_df[feature].min()
                max_val = data_df[feature].max()
        else:
            has_nonpositive = (data_df[feature] <= 0).any()
            if has_nonpositive:
                num_negative = (data_df[feature] <= 0).sum()
                min_val = data_df[feature].min()
                max_val = data_df[feature].max()
                
        if has_nonpositive:
            print(f"⚠️  Skipping histogram for '{feature}': Cannot use log scale with {num_negative} non-positive values (log requires all positive values).")
            print(f"   Min value: {min_val}, Max value: {max_val}")
            print(f"   Consider using log_scale=False or filtering data to positive values only.")
            plt.close('all')  # Clean up any matplotlib artifacts
            return None
            
        data_df = _create_log_column(data_df, feature, new_feature, log_base)
        feature_to_plot = new_feature
        x_label = f"log{log_base}({feature})" if log_axis_label != "actual" else feature
    else:
        feature_to_plot = feature
        x_label = feature

    figure_background = (
        "none" if transparent else background if is_color_like(background) else "white"
    )
    axis_background = figure_background

    if separate is not None:
        unique_categories = _get_unique_categories(data_df, separate, order)
    else:
        unique_categories = [None]

    color_palette = _process_palette(palette, len(unique_categories))

    # Ensure condition_colors is not None
    if condition_colors is None:
        condition_colors = {}

    color_mapping = {}
    for i, category in enumerate(unique_categories):
        if category in condition_colors:
            color_mapping[category] = condition_colors[category]
        else:
            color_mapping[category] = color_palette[i % len(color_palette)]

    if x_range is not None:
        global_lower_bound, global_upper_bound = x_range
    else:
        global_lower_bound = _column_min(data_df, feature_to_plot)
        global_upper_bound = (
            xlimit if xlimit is not None else _column_max(data_df, feature_to_plot)
        )

    # fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if small_multiples and separate is not None:
        fig, axes = plt.subplots(
            len(unique_categories),
            1,
            figsize=(figsize[0], figsize[1] * len(unique_categories)),
            sharex=True,
            facecolor=figure_background,
        )
        if len(unique_categories) == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
        axes = [ax]

    for i, category in enumerate(unique_categories):
        ax = axes[i] if (small_multiples and separate) else axes[0]
        
        # Get subset values
        if category:
            subset_df = _filter_dataframe(data_df, separate, category)
            if is_polars_df:
                subsetvalues_col = subset_df[feature_to_plot]
            else:
                subsetvalues_col = subset_df[feature_to_plot]
        else:
            subsetvalues_col = data_df[feature_to_plot]
        
        # Filter by range
        if is_polars_df:
            filtered_df = _filter_range(
                pl.DataFrame({feature_to_plot: subsetvalues_col}),
                feature_to_plot,
                global_lower_bound,
                global_upper_bound
            )
            subsetvalues = _to_numpy(filtered_df[feature_to_plot])
        else:
            mask = (subsetvalues_col >= global_lower_bound) & (subsetvalues_col <= global_upper_bound)
            subsetvalues = subsetvalues_col[mask].values

        ax.set_facecolor(axis_background)
        bin_edges = np.linspace(global_lower_bound, global_upper_bound, bins + 1)

        if use_kde:
            sns.kdeplot(
                subsetvalues,
                fill=kde_fill,
                ax=ax,
                color=color_mapping[category],
                linewidth=1.5,
                label=category,
                alpha=alpha,
            )
        else:
            counts, _ = np.histogram(subsetvalues, bins=bin_edges)
            if percentage:
                counts = 100 * counts / counts.sum() if counts.sum() > 0 else counts
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(
                bin_centers,
                counts,
                width=np.diff(bin_edges),
                color=color_mapping[category],
                alpha=alpha,
                label=category,
            )

        # Calculate average
        if average == "mean":
            avg_value = np.mean(subsetvalues)
        else:
            avg_value = np.median(subsetvalues)
            
        annotation_value = (
            log_base**avg_value
            if log_scale and log_axis_label == "actual"
            else avg_value
        )

        if showavg:
            ax.axvline(avg_value, color=line_color, linestyle="--")

        # Apply line_color styling to each subplot individually
        ax.set_xlabel(x_label, fontsize=scaled_font, color=line_color)
        ax.set_ylabel(
            "Percentage" if percentage else "Count", fontsize=scaled_font, color=line_color
        )
        ax.tick_params(
            axis="both",
            which="both",
            color=line_color,
            labelcolor=line_color,
            labelsize=scaled_font,
        )

        if grid:
            ax.grid(
                True,
                linestyle="--",
                linewidth=0.5,
                color=line_color if not transparent else (0, 0, 0, 0.5),
                alpha=0.7,
                axis="y",
            )

        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
            if transparent:
                spine.set_alpha(0.9)

        ax.set_xlim(global_lower_bound, global_upper_bound)
        xticks = np.arange(
            global_lower_bound, global_upper_bound + tick_interval, tick_interval
        )
        ax.set_xticks(xticks)
        if y_range is not None:
            ax.set_ylim(y_range)
        if log_scale:
            ax.set_xscale("linear")
            if log_axis_label == "actual":
                formatter = FuncFormatter(lambda val, pos: f"{log_base ** val:.2g}")
                ax.xaxis.set_major_formatter(formatter)

    # legend = ax.legend(title=separate, fontsize=scaled_font, title_fontsize=scaled_font, loc='upper left', bbox_to_anchor=(1, 1))
    # plt.gca().add_artist(legend)
    if small_multiples:
        fig.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper right",
            bbox_to_anchor=(1, 1),
        )
    else:
        legend = ax.legend(
            title=separate,
            fontsize=scaled_font,
            title_fontsize=scaled_font,
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        plt.gca().add_artist(legend)

    if save_folder is None:
        save_folder = os.path.join(master_dir, "plots", "histograms")
    os.makedirs(save_folder, exist_ok=True)
    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    out_filename = f"{feature}_histogram.{ext}"
    full_save_path = os.path.join(save_folder, out_filename)
    plt.savefig(
        full_save_path, bbox_inches="tight", transparent=transparent, format=ext
    )

    svg_data = None
    if ext == "svg":
        with open(full_save_path, encoding="utf-8") as f:
            svg_data = f.read()
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\([^)]*\)"', "", svg_data)
        svg_data = re.sub(r"<metadata>.*?</metadata>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<\?xml[^>]*\?>", "", svg_data, flags=re.DOTALL)
        svg_data = re.sub(r"<!DOCTYPE[^>]*>", "", svg_data, flags=re.DOTALL)
        svg_data = svg_data.strip()
        with open(full_save_path, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


def plot_boxplots_svg(
    data_df,
    feature,
    x_category,
    font_size=12,
    order=None,
    palette="colorblind",
    background="white",
    transparent=False,
    line_color="black",
    show_plot=True,
    master_dir=None,
    grid=True,
    bw=False,
    strip=False,
    y_max=None,
    y_min=None,
    figsize=(10, 8),
    annotate_median=False,
    rotation=90,
    dotsize=3,
    custom="_",
    export_format="png",
    return_svg=False,
    annotatemultiplier=0.95,
    annotation_decimals=2,
    annotate_stat="median",
):
    """
    Plot boxplots for the specified feature against a categorical x_category with custom order and styling options.

    Compatible with both Polars and Pandas DataFrames.

    Parameters
    ----------
    data_df : DataFrame (Polars or Pandas)
        DataFrame containing the data.
    feature : str
        The feature to plot on the y-axis.
    x_category : str
        The categorical feature to plot on the x-axis.
    font_size : int, optional
        Font size for the plot text. Default is 12.
    order : list, optional
        Specific order for the categories. Default is None.
    palette : str or list, optional
        Color palette for the plot. Can be:
        - String: palette name ('colorblind' uses Wong 2011, 'Dark2', 'Set2', etc.)
        - List: custom colors as hex codes, named colors, or RGB tuples
        Example: ['#0173B2', '#DE8F05', '#029E73'] or ['red', 'blue', 'green']
        Default is 'colorblind'.
    background : str, tuple, optional
        Background color as a color name, RGB tuple, or hex code. Default is 'white'.
    transparent : bool, optional
        If True, makes the figure background transparent. Default is False.
    line_color : str, optional
        Color of all plot lines, including box edges, axis lines, grid lines, labels, and strip plot dots. Default is 'black'.
    show_plot : bool, optional
        Whether to display the plot in the notebook. Default is True.
    master_dir : str, optional
        Directory to save the plot. If not provided, defaults to "plots".
    grid : bool, optional
        Whether to display grid lines. Default is True.
    bw : bool, optional
        Whether to use black-and-white styling. Default is False.
    strip : bool, optional
        Whether to overlay a stripplot on the boxplot. Default is False.
    y_max : float, optional
        Maximum value for the y-axis. Default is None.
    figsize : tuple, optional
        Size of the plot (width, height) in inches. Default is (10, 8).
    annotate_median : bool, optional
        If True, annotate each boxplot with its median value. Default is False.
    rotation : int, optional
        Rotation angle for tick labels. Default is 90.
    dotsize : int, optional
        Size of the dots in the strip plot. Default is 3.
    export_format : str, optional
        File format to export the figure ('png' or 'svg'). Default is 'png'.
    return_svg : bool, optional
        If True and export_format is 'svg', returns the post-processed SVG image data as a string.
    annotation_decimals : int, optional
        Number of decimal places to display for annotated values. Default is 2.
    annotate_stat : {"median", "mean"}, optional
        Statistic written above each box when annotate_median is True. Default is "median".

    Returns
    -------
    str or None
        When export_format is 'svg' and return_svg is True, returns the cleaned SVG data as a string.
        Otherwise, returns None.

    """
    # Detect dataframe type and convert if needed
    is_polars_df = _is_polars(data_df)
    
    # For seaborn plotting, we need pandas DataFrame
    # Store the original for median calculation if using polars
    original_df = data_df
    if is_polars_df:
        # Convert to pandas for seaborn - only select needed columns for efficiency
        plot_df = data_df.select([x_category, feature]).to_pandas()
    else:
        plot_df = data_df
    
    # Validate and apply background color
    if is_color_like(background):
        figure_background = background
    else:
        print("Invalid color provided for background. Defaulting to white.")
        figure_background = "white"

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=figsize, facecolor=figure_background)
    if not transparent:
        ax.set_facecolor(figure_background)
    else:
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))

    sns.set_context(
        "notebook", rc={"xtick.labelsize": font_size, "ytick.labelsize": font_size}
    )

    # Get number of categories
    if order is not None:
        n_categories = len(order)
    else:
        if is_polars_df:
            n_categories = original_df[x_category].n_unique()
        else:
            n_categories = len(plot_df[x_category].unique())
    
    # Process palette to handle both string names and custom color lists
    processed_palette = _process_palette(palette, n_categories)

    if bw:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=plot_df,
            linewidth=1.5,
            showfliers=False,
            color="white",
            order=order,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for element in ["boxes", "whiskers", "medians", "caps"]:
            plt.setp(boxplot.artists, color=line_color)
            plt.setp(boxplot.lines, color=line_color)
    else:
        boxplot = sns.boxplot(
            x=x_category,
            y=feature,
            data=plot_df,
            palette=processed_palette,
            order=order,
            showfliers=False,
            linewidth=1.5,
        )
        for patch in boxplot.patches:
            patch.set_edgecolor(line_color)
            patch.set_linewidth(1.5)
        for line in boxplot.lines:
            line.set_color(line_color)
            line.set_linewidth(1.5)

    if strip:
        sns.stripplot(
            x=x_category,
            y=feature,
            data=plot_df,
            color=line_color,
            size=dotsize,
            order=order,
            jitter=True,
        )

    plt.xlabel("", fontsize=font_size, color=line_color)
    plt.ylabel(feature, fontsize=font_size, color=line_color)
    ax.tick_params(
        axis="both",
        which="both",
        color=line_color,
        labelcolor=line_color,
        labelsize=font_size,
        rotation=rotation,
    )

    if grid:
        ax.grid(
            True, linestyle="--", linewidth=0.5, color=line_color, alpha=0.7, axis="y"
        )

    if y_max is not None:
        plt.ylim(top=y_max)

    if y_min is not None:
        plt.ylim(bottom=y_min)

    for spine in ax.spines.values():
        spine.set_edgecolor(line_color)
        spine.set_linewidth(1.5)
        if transparent:
            spine.set_alpha(0.5)

    if annotate_median:
        # Keep labels aligned to the plotted category positions.
        # Skipping missing categories used to shift later annotations onto the wrong boxes.
        stat_name = annotate_stat if annotate_stat in {"mean", "median"} else "median"
        stats_map = _groupby_agg(original_df, x_category, feature, stat_name)
        if hasattr(stats_map, "to_dict"):
            stats_map = stats_map.to_dict()
        current_y_max = plt.ylim()[1]
        categories = list(order) if order else list(stats_map.keys())
        for i, cat in enumerate(categories):
            value = stats_map.get(cat)
            if value is None or pd.isna(value):
                continue
            plt.text(
                i,
                current_y_max * annotatemultiplier,
                f"{value:.{annotation_decimals}f}",
                horizontalalignment="center",
                size=font_size,
                color=line_color,
                weight="bold",
            )

    plt.tight_layout()

    if master_dir is None:
        master_dir = "plots"
    os.makedirs(master_dir, exist_ok=True)

    # condsindf = data_df['condition'].unique()

    ext = export_format.lower()
    if ext not in ["png", "svg"]:
        print("Invalid export format specified. Defaulting to 'png'.")
        ext = "png"
    filename = f"{master_dir}/{feature}_by_{x_category}_{custom}.{ext}"

    # Save the figure to file in the requested format
    plt.savefig(filename, bbox_inches="tight", transparent=transparent, format=ext)

    svg_data = None
    # If exporting to SVG, post-process the file to remove clipping paths.
    if ext == "svg":
        with open(filename, encoding="utf-8") as f:
            svg_data = f.read()
        # Remove any <clipPath> definitions and clip-path attributes.
        svg_data = re.sub(
            r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL
        )
        svg_data = re.sub(r'\s*clip-path="url\(#.*?\)"', "", svg_data)
        # Write the cleaned SVG back to the file.
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_data)

    if show_plot:
        plt.show()
    else:
        plt.close()

    if ext == "svg" and return_svg:
        return svg_data


def plot_boxplot_with_file_medians(
    data_df,
    feature,
    x_category,
    order,
    master_dir,
    per_file_col="filename",
    y_limits=(0, None),
    figsize=(2.8, 3.1),
    annotate_median=True,
    dot_size=2,
    font_size=14,
    annotation_font_size=10,
    line_color="k",
    annotate_multiplier=0.93,
    annotation_decimals=2,
    show_plot=True,
):
    """Boxes from every window, dots = one median per cell (per_file_col). Annotation = median of all windows.

    Saves {feature}_by_{x_category}_allwindows_{per_file_col}_median_dots.svg and returns a count table.
    """
    if isinstance(data_df, pl.DataFrame):
        data_df = data_df.select([per_file_col, x_category, feature]).to_pandas()
    box_df = data_df[[per_file_col, x_category, feature]].dropna().copy()
    box_df[x_category] = box_df[x_category].astype(str)
    plot_order = [c for c in order if c in set(box_df[x_category])]
    box_df = box_df[box_df[x_category].isin(plot_order)]
    dot_df = box_df.groupby([per_file_col, x_category], as_index=False)[feature].median()
    palette = sns.color_palette("colorblind", len(plot_order))
    lc = line_color

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))

    sns.boxplot(
        data=box_df, x=x_category, y=feature, hue=x_category, order=plot_order, hue_order=plot_order,
        palette=palette, legend=False, showfliers=False, linewidth=1.5, ax=ax,
    )
    for patch in ax.patches:
        patch.set_edgecolor(lc)
        patch.set_linewidth(1.5)
    for line in ax.lines:
        line.set_color(lc)
        line.set_linewidth(1.5)

    sns.stripplot(
        data=dot_df, x=x_category, y=feature, order=plot_order,
        color=lc, size=dot_size, jitter=True, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(feature, fontsize=font_size, color=lc)
    ax.tick_params(axis="both", color=lc, labelcolor=lc, labelsize=font_size)
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, linestyle="--", linewidth=0.5, color=lc, alpha=0.7, axis="y")
    ax.grid(False, axis="x")

    y_min, y_max = y_limits
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    for spine in ax.spines.values():
        spine.set_edgecolor(lc)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)

    if annotate_median:
        window_medians = box_df.groupby(x_category)[feature].median()
        y_text = ax.get_ylim()[1] * annotate_multiplier
        for i, cat in enumerate(plot_order):
            ax.text(i, y_text, f"{window_medians[cat]:.{annotation_decimals}f}", ha="center",
                    size=annotation_font_size, color=lc, weight="bold")

    fig.tight_layout()
    os.makedirs(master_dir, exist_ok=True)
    out_path = os.path.join(master_dir, f"{feature}_by_{x_category}_allwindows_{per_file_col}_median_dots.svg")
    fig.savefig(out_path, bbox_inches="tight", transparent=True, format="svg")
    with open(out_path, encoding="utf-8") as f:
        svg_data = f.read()
    svg_data = re.sub(r'<clipPath id="[^"]*">.*?</clipPath>', "", svg_data, flags=re.DOTALL)
    svg_data = re.sub(r'\s*clip-path="url\(#.*?\)"', "", svg_data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg_data)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    counts = pd.DataFrame({
        "n_windows": box_df.groupby(x_category).size(),
        f"n_{per_file_col}": dot_df.groupby(x_category).size(),
        "median_all_windows": box_df.groupby(x_category)[feature].median(),
        f"median_of_{per_file_col}_medians": dot_df.groupby(x_category)[feature].median(),
    }).reindex(plot_order)
    print(f"{feature}: saved {out_path}")
    return counts
