"""Only executable basenames are matched; names never enter the observation store."""

from __future__ import annotations

from pathlib import PureWindowsPath

CATEGORIES = ("coding", "browsing", "media", "compute", "other")
DEFAULT_MAPPINGS = {
    "code": "coding",
    "pycharm64": "coding",
    "idea64": "coding",
    "devenv": "coding",
    "eclipse": "coding",
    "sublime_text": "coding",
    "chrome": "browsing",
    "firefox": "browsing",
    "msedge": "browsing",
    "brave": "browsing",
    "vlc": "media",
    "spotify": "media",
    "wmplayer": "media",
    "mpv": "media",
    "blender": "compute",
    "handbrake": "compute",
    "ffmpeg": "compute",
}


def validate_mappings(mapping: dict[str, str]) -> dict[str, str]:
    if not isinstance(mapping, dict) or len(mapping) > 1000:
        raise ValueError("Mappings must be a small object of executable basenames to categories")
    result = {}
    for name, category in mapping.items():
        if not isinstance(name, str) or any(c in name for c in ("/", "\\", "\n", "\r")):
            raise ValueError("Use executable basenames only, never paths")
        if not name.strip() or len(name) > 100 or category not in CATEGORIES:
            raise ValueError("Invalid executable/category mapping")
        result[name.lower().removesuffix(".exe")] = category
    return result


def categorize(name: str | None, mappings: dict[str, str]) -> str:
    if not name:
        return "other"
    return mappings.get(PureWindowsPath(name).name.lower().removesuffix(".exe"), "other")
