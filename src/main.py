import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import ctypes
import os
import sys
import config
from enhance import load_model

if os.name == "nt":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MoFiCo.OpenMFC.1.0")


# ─────────────────────────────────────────
#  Asset path helper
# ─────────────────────────────────────────

def _asset_path(relative: str) -> str:
    """Resolve path to bundled asset — works both in dev and PyInstaller exe."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ─────────────────────────────────────────
#  Icon generation (run once)
# ─────────────────────────────────────────

def ensure_icon() -> None:
    ico_path = _asset_path(os.path.join("assets", "icon.ico"))
    png_path = _asset_path(os.path.join("assets", "OpenMFC ICON.png"))
    if not os.path.exists(ico_path) and os.path.exists(png_path):
        try:
            from PIL import Image
            os.makedirs(os.path.dirname(ico_path), exist_ok=True)
            img = Image.open(png_path)
            img.save(ico_path, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
            print("  Icon generated:", ico_path)
        except Exception as e:
            print(f"  Icon generation skipped: {e}")


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────

def main() -> None:
    print("OpenMFC starting...")

    ensure_icon()

    # Resolve model path relative to bundle root
    config.ONNX_MODEL_PATH = _asset_path(config.ONNX_MODEL_PATH)

    onnx_session = None
    if config.ENABLE_ONNX:
        onnx_session = load_model(config.ONNX_MODEL_PATH)
        
    # Open the window immediately with no camera list — enumeration runs
    # in a background thread after the UI is up to avoid the startup spike
    # from probing up to 10 VideoCapture indices synchronously.
    from ui import build_ui
    build_ui(onnx_session=onnx_session, camera_labels=None)


if __name__ == "__main__":
    main()