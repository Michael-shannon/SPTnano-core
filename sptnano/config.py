"""Read the yaml files in configs/ and fix paths for Windows or WSL."""

from __future__ import annotations

import platform
from pathlib import Path

import yaml

REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "configs"


def running_on_linux() -> bool:
    return platform.system() == "Linux"


def to_wsl_path(path) -> str:
    """D:/data -> /mnt/d/data. Leaves Linux paths alone."""
    path = str(path).replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        return f"/mnt/{path[0].lower()}{path[2:]}"
    return path


def to_windows_path(path) -> str:
    """/mnt/d/data -> D:/data. Leaves Windows paths alone."""
    path = str(path).replace("\\", "/")
    if path.startswith("/mnt/") and len(path) > 6:
        return f"{path[5].upper()}:/{path[7:]}"
    return path


def fix_path(path) -> Path:
    """Make a stored path usable on this machine (Windows or WSL), relative to the repo if not absolute."""
    path = to_wsl_path(path) if running_on_linux() else to_windows_path(path)
    path = Path(path)
    if not path.is_absolute() and not str(path).startswith("/"):
        path = REPO_DIR / path
    return path


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_paths() -> dict[str, Path]:
    """Paths from configs/paths.local.yaml (falls back to paths.example.yaml)."""
    name = "paths.local.yaml" if (CONFIG_DIR / "paths.local.yaml").exists() else "paths.example.yaml"
    if name == "paths.example.yaml":
        print("configs/paths.local.yaml not found - using paths.example.yaml. Copy it and edit your folders.")
    return {key: fix_path(value) for key, value in load_yaml(name).items() if value is not None}


def load_settings() -> dict:
    return load_yaml("pipeline.yaml")


def load_gates() -> dict:
    return load_yaml("gates.yaml")


# Defaults for the plotting functions, from configs/pipeline.yaml.
_settings = load_settings()
PIXELSIZE_MICRONS = _settings["acquisition"]["pixel_size_um"]
TIME_BETWEEN_FRAMES = _settings["acquisition"]["frame_interval_s"]
TIME_WINDOW = _settings["windows"]["window_size"]
OVERLAP = _settings["windows"]["overlap"]
MASTER = None
