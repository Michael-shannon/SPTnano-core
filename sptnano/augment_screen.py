"""Recompute window features from x/y, used by the augmentation screen in notebook 04."""

import numpy as np
import polars as pl

def parse_window_uid(wuid):
    """
    Parse a window UID string to extract unique ID, start frame, and end frame.
    
    Args:
        wuid: Window UID string in format 'unique_id_start_frame_end_frame'
    
    Returns:
        tuple: (unique_id, start_frame, end_frame)
    """
    parts = wuid.rsplit('_', 2)
    return '_'.join(parts[:-2]).rsplit('_', 1)[0], int(float(parts[-2])), int(float(parts[-1]))


def get_window_frames(df, wuid):
    """
    Extract frames for a specific window UID from a dataframe.
    
    Args:
        df: Polars dataframe with 'unique_id' and 'frame' columns
        wuid: Window UID string
    
    Returns:
        Polars dataframe filtered to the specified window, sorted by frame
    """
    uid, fs, fe = parse_window_uid(wuid)
    return df.filter(
        (pl.col('unique_id') == uid) & 
        (pl.col('frame') >= fs) & 
        (pl.col('frame') <= fe)
    ).sort('frame')


def calc_intersections(x, y):
    """
    Calculate the number of self-intersections in a track.
    
    Uses the cross-product method to detect line segment intersections.
    
    Args:
        x, y: Track coordinates (numpy arrays)
    
    Returns:
        int: Number of self-intersections
    """
    n = len(x)
    if n < 4: 
        return 0
    
    pts = np.column_stack([x, y])
    
    # Counter-clockwise check
    ccw = lambda a, b, c: (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    # Intersection check
    intersect = lambda a, b, c, d: ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    
    return sum(
        1 for i in range(n - 1) 
        for j in range(i + 2, n - 1) 
        if intersect(pts[i], pts[i + 1], pts[j], pts[j + 1])
    )


def compute_features(x, y, dt_arr):
    """
    Compute track features: average speed, cumulative displacement, 
    self-intersections, and anomalous exponent.
    
    Args:
        x, y: Track coordinates (numpy arrays)
        dt_arr: Time deltas (can be scalar or array)
    
    Returns:
        dict: Dictionary with keys:
            - 'avg_speed_um_s': Average speed in um/s
            - 'cum_displacement_um': Cumulative displacement in um
            - 'self_intersections': Number of self-intersections
            - 'anomalous_exponent': Anomalous diffusion exponent (alpha)
    """
    import scipy.optimize
    
    dx, dy = np.diff(x), np.diff(y)
    steps = np.sqrt(dx**2 + dy**2)
    
    # Handle time deltas
    dta = np.asarray(dt_arr)[1:] if not np.isscalar(dt_arr) else dt_arr
    if not np.isscalar(dt_arr):
        dta = np.where((dta == 0) | np.isnan(dta), 1e-10, dta)
    
    speeds = steps / dta if np.isscalar(dta) else steps / dta
    dt = dt_arr if np.isscalar(dt_arr) else np.nanmedian(dta)
    
    n = len(x)
    lags = np.arange(1, n) * dt
    
    # Mean squared displacement
    msd = np.array([
        np.mean((x[l:] - x[:-l])**2 + (y[l:] - y[:-l])**2) 
        for l in range(1, n)
    ])
    
    # Fit anomalous exponent: MSD = 4*D*t^alpha
    try:
        popt, _ = scipy.optimize.curve_fit(
            lambda t, D, a: 4 * D * np.power(t, a),
            lags,
            msd,
            p0=[0.1, 1],
            bounds=([0, 0], [np.inf, 3]),
            maxfev=5000
        )
        alpha = popt[1]
    except:
        # Fallback to log-log linear fit
        v = (lags > 0) & (msd > 0)
        if np.sum(v) >= 2:
            alpha = np.polyfit(np.log(lags[v]), np.log(msd[v]), 1)[0]
        else:
            alpha = 1.0
    
    return {
        'avg_speed_um_s': np.mean(speeds),
        'cum_displacement_um': np.sum(steps),
        'self_intersections': calc_intersections(x, y),
        'anomalous_exponent': alpha
    }


def has_complete_frames(df, wuid):
    """
    Check if a track has all expected frames (no missing frames).
    
    Args:
        df: Polars dataframe with 'unique_id' and 'frame' columns
        wuid: Window UID string
    
    Returns:
        bool: True if track has all expected frames, False otherwise
    """
    uid, fs, fe = parse_window_uid(wuid)
    expected_frames = fe - fs + 1
    actual_frames = df.filter(
        (pl.col('unique_id') == uid) & 
        (pl.col('frame') >= fs) & 
        (pl.col('frame') <= fe)
    ).height
    return actual_frames == expected_frames
