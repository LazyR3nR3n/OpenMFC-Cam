import cv2
import platform
import subprocess
from typing import Optional


# --- Internal cache ---

_camera_list: list[dict] = []  # [{"index": 0, "name": "Camera 0 (Built-in)"}]


# --- Platform-aware name fetching ---

def _get_camera_names_windows() -> dict[int, str]:
    """Try to get friendly camera names on Windows via PowerShell."""
    names = {}
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-PnpDevice -Class Camera | Select-Object -ExpandProperty FriendlyName",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        for i, name in enumerate(lines):
            names[i] = name
    except Exception:
        pass
    return names


def _get_camera_names_linux() -> dict[int, str]:
    """Try to get friendly camera names on Linux via v4l2."""
    names = {}
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        current_name = None
        index = 0
        for line in lines:
            if not line.startswith("\t"):
                current_name = line.split("(")[0].strip()
            else:
                dev = line.strip()
                if dev.startswith("/dev/video"):
                    try:
                        idx = int(dev.replace("/dev/video", ""))
                        names[idx] = current_name or f"Camera {index}"
                        index += 1
                    except ValueError:
                        pass
    except Exception:
        pass
    return names


def _get_platform_names() -> dict[int, str]:
    system = platform.system()
    if system == "Windows":
        return _get_camera_names_windows()
    elif system == "Linux":
        return _get_camera_names_linux()
    return {}


# --- Core enumeration ---

def enumerate_cameras(max_index: int = 10) -> list[dict]:
    """
    Probe camera indices 0 through max_index-1.
    Returns a list of dicts: [{"index": int, "name": str}]
    Caches result in _camera_list.
    """
    global _camera_list
    _camera_list = []

    platform_names = _get_platform_names()

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2)
        if not cap.isOpened():
            # Try generic backend as fallback
            cap = cv2.VideoCapture(i)

        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                name = platform_names.get(i, f"Camera {i}")
                _camera_list.append({"index": i, "name": name})

    if not _camera_list:
        # Fallback: assume at least index 0 exists
        _camera_list.append({"index": 0, "name": "Camera 0 (default)"})

    return _camera_list


# --- Accessors ---

def get_camera_list() -> list[dict]:
    """Return cached camera list. Call enumerate_cameras() first."""
    return _camera_list


def get_camera_labels() -> list[str]:
    """Return display labels for UI dropdown."""
    return [f"{cam['name']} (index {cam['index']})" for cam in _camera_list]


def get_index_from_label(label: str) -> int:
    """Parse camera index from a dropdown label string."""
    for cam in _camera_list:
        if f"(index {cam['index']})" in label:
            return cam["index"]
    return 0


def get_default_camera() -> Optional[dict]:
    """Return the first detected camera, or None if none found."""
    return _camera_list[0] if _camera_list else None