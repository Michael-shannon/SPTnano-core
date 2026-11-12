"""Reading raw TIFF movies and Fiji LUTs (shared by movies.py and contours.py)."""

import numpy as np
import os
import pandas as pd
import polars as pl
import skimage.io as skio
from matplotlib.colors import LinearSegmentedColormap

def _napari_colormap(rgba, name):
    try:
        from napari.utils.colormaps import Colormap
    except ImportError:
        return None
    return Colormap(rgba, name=name)


def _is_napari_colormap(obj):
    try:
        from napari.utils.colormaps import Colormap
    except ImportError:
        return False
    return isinstance(obj, Colormap)


def load_image(file_path):
    return skio.imread(file_path)  # changed to skio to avoid conflict with imageio


def load_lut_from_file(path):
    """
    Load a Fiji LUT file and return a (256, 4) float32 RGBA array for Napari.
    """
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    if raw.size >= 768:
        raw = raw[-768:] if raw.size != 768 else raw
        r, g, b = raw[:256], raw[256:512], raw[512:768]
    elif raw.size == 256:
        r = g = b = raw
    else:
        raise ValueError(f"Unsupported Fiji LUT size for {path!r}: {raw.size} bytes")
    rgba_uint8 = np.stack([r, g, b, np.full_like(r, 255)], axis=1)
    rgba_float = rgba_uint8.astype(np.float32) / 255.0
    return rgba_float


def fiji_lut_to_python(lut_path, name=None):
    """
    Convert one Fiji LUT into reusable Python colormaps.

    Returns a dict with the LUT path, RGBA array, a napari ``Colormap``, and a
    matplotlib ``LinearSegmentedColormap``.
    """
    rgba = load_lut_from_file(lut_path)
    if name is None:
        name = os.path.splitext(os.path.basename(lut_path))[0]
    safe_name = str(name).replace(" ", "_")
    return {
        "name": name,
        "path": str(lut_path),
        "rgba": rgba,
        "napari": _napari_colormap(rgba, name=safe_name),
        "matplotlib": LinearSegmentedColormap.from_list(safe_name, rgba),
    }


def load_fiji_lut_collection(
    lut_dir=r"D:\GITHUB_SOFTWARE\Fiji.app\luts",
    *,
    prefix="JDM",
    default_lut_name="JDM_Ink Wash Yellow",
):
    """
    Load all Fiji LUTs with a given prefix and return ``(lut_dict, chosen_LUT)``.

    ``chosen_LUT`` is the requested default when available; otherwise the first
    matching LUT is used. This is intended for notebook use:

    ``jdm_luts, chosen_LUT = load_fiji_lut_collection(...)``
    """
    lut_paths = []
    if os.path.isdir(lut_dir):
        for name in sorted(os.listdir(lut_dir)):
            if name.lower().endswith(".lut") and name.startswith(prefix):
                lut_paths.append(os.path.join(lut_dir, name))

    if len(lut_paths) == 0:
        raise FileNotFoundError(f"No {prefix!r} LUT files found in {lut_dir!r}")

    lut_dict = {}
    for path in lut_paths:
        lut_name = os.path.splitext(os.path.basename(path))[0]
        lut_dict[lut_name] = fiji_lut_to_python(path, name=lut_name)

    if default_lut_name in lut_dict:
        chosen_LUT = lut_dict[default_lut_name]
    else:
        chosen_name = sorted(lut_dict)[0]
        print(
            f"[LUT] Requested {default_lut_name!r} was not found. "
            f"Using {chosen_name!r} instead."
        )
        chosen_LUT = lut_dict[chosen_name]

    print(f"[LUT] Loaded {len(lut_dict)} {prefix!r} LUTs from {lut_dir}")
    print(f"[LUT] chosen_LUT: {chosen_LUT['name']}")
    return lut_dict, chosen_LUT


def _coerce_lut_colormap(lut=None, *, target="matplotlib", fallback="gray"):
    if lut is None:
        return fallback
    if isinstance(lut, dict):
        if target in lut:
            return lut[target]
        if "rgba" in lut:
            name = lut.get("name", "fiji_lut").replace(" ", "_")
            if target == "napari":
                return _napari_colormap(lut["rgba"], name=name)
            return LinearSegmentedColormap.from_list(name, lut["rgba"])
    if target == "napari" and _is_napari_colormap(lut):
        return lut
    return lut


def _as_pandas_copy(df):
    """Return a pandas copy while accepting pandas or Polars input."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    raise TypeError("Expected a pandas.DataFrame or polars.DataFrame")


def _single_value_from_df(df, column, default=None):
    if column not in df.columns or df.empty:
        return default
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return default
    return values[0]


def _resolve_raw_movie_path(single_cell_df, master_dir=None, filename=None, condition=None):
    df = _as_pandas_copy(single_cell_df)
    filename = filename or _single_value_from_df(df, "filename")
    condition = condition or _single_value_from_df(df, "condition")
    if filename is None:
        raise ValueError("Could not infer filename from single_cell_df")
    image_filename = str(filename).replace("_tracked", "") + ".tif"

    if master_dir is None:
        master_dir = _single_value_from_df(df, "foldername")
    if master_dir is None:
        raise ValueError("master_dir was not provided and foldername was not found")

    candidates = []
    if condition is not None:
        candidates.append(os.path.join(master_dir, "data", str(condition), image_filename))
        candidates.append(os.path.join(master_dir, str(condition), image_filename))
    candidates.append(os.path.join(master_dir, "data", image_filename))
    candidates.append(os.path.join(master_dir, image_filename))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return candidates[0]


def _normalize_raw_image(image, contrast_limits=None, invert_image=False):
    image = image.astype(float, copy=False)
    if contrast_limits is None:
        lower = np.nanmin(image)
        upper = np.nanmax(image)
    else:
        lower, upper = contrast_limits
    if upper <= lower:
        norm = np.zeros_like(image, dtype=float)
    else:
        norm = np.clip((image - lower) / (upper - lower), 0, 1)
    if invert_image:
        norm = 1 - norm
    return norm


def _load_raw_frame_indices(raw_movie_path, frame_indices):
    """
    Load only requested frames for TIFF stacks when possible.

    Falls back to the project ``load_image`` helper for formats/readers that do
    not support partial TIFF reads.
    """
    frame_indices = np.asarray(frame_indices, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("No frame indices requested")
    try:
        import tifffile

        with tifffile.TiffFile(raw_movie_path) as tif:
            image = tif.asarray(key=frame_indices.tolist())
    except Exception:
        full_image = load_image(raw_movie_path)
        image = full_image[frame_indices]

    image = np.asarray(image)
    if image.ndim == 2 and frame_indices.size > 1:
        image = image[np.newaxis, ...]
    return image
