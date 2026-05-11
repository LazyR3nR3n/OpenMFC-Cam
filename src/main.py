import os
import config
import ctypes
from devices import enumerate_cameras, get_camera_labels
from enhance import load_model
import sys
import ctypes

ctypes.windll.kernel32.FreeConsole()

if sys.platform == "win32":
    import ctypes
    ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

if os.name == "nt":
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MoFiCo.OpenMFC.1.0")

# ─────────────────────────────────────────
#  Icon generation (run once)
# ─────────────────────────────────────────

def ensure_icon() -> None:
    ico_path = os.path.join("assets", "icon.ico")
    png_path = os.path.join("assets", "OpenMFC ICON.png")
    if not os.path.exists(ico_path) and os.path.exists(png_path):
        try:
            from PIL import Image
            os.makedirs("assets", exist_ok=True)
            img = Image.open(png_path)
            img.save(ico_path, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)])
            print("  Icon generated: assets/icon.ico")
        except Exception as e:
            print(f"  Icon generation skipped: {e}")


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────

def main() -> None:
    global _onnx_session, _camera_labels

    print("OpenMFC starting...")

    ensure_icon()

    print("Detecting cameras...")
    cameras = enumerate_cameras()
    camera_labels = get_camera_labels()
    if cameras:
        config.CAPTURE_DEVICE = cameras[0]["index"]
        print(f"  Found {len(cameras)} camera(s). Using: {cameras[0]['name']}")
    else:
        print("  No cameras found. Defaulting to index 0.")

    onnx_session = None
    if config.ENABLE_ONNX:
        onnx_session = load_model(config.ONNX_MODEL_PATH)

    from ui import build_ui
    build_ui(onnx_session=onnx_session, camera_labels=camera_labels)


if __name__ == "__main__":
    main()