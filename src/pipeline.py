import cv2
import os
import tempfile
import threading
import time
import config
import numpy as np
from pathlib import Path
from mfnr import load_burst, mfnr, save_output
from hdr import run_hdr, save_hdr_output
from enhance import run_enhance, save_enhance_output


# ─────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────

def run_pipeline(
    image_paths: list[str],
    onnx_session,
    log_fn,
    on_done_fn=None,
    preview_callback=None,
) -> None:
    log_fn(f"[pipeline] Starting with {len(image_paths)} frames...")
    images = load_burst(image_paths)
    if not images:
        log_fn("[pipeline] No valid images loaded. Aborting.")
        return

    log_fn("[pipeline] Running MFNR (Stage 1)...")
    mfnr_result = mfnr(images, ecc_threshold=config.ECC_THRESHOLD)
    if mfnr_result is None:
        log_fn("[pipeline] MFNR returned None. Aborting.")
        return

    out_path = Path(config.OUTPUT_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    if config.ENABLE_HDR:
        log_fn("[pipeline] Running HDR fusion (Stage 2)...")
        hdr_result = run_hdr(mfnr_result)
        enhance_input = hdr_result
    else:
        log_fn("[pipeline] HDR disabled, skipping Stage 2.")
        enhance_input = cv2.normalize(
            mfnr_result, None, 0, 255, cv2.NORM_MINMAX
        ).astype("uint8")

    if config.ENABLE_ONNX:
        log_fn("[pipeline] Running enhancement (Stage 3)...")
        final = run_enhance(enhance_input, session=onnx_session)
    else:
        log_fn("[pipeline] ONNX disabled, skipping Stage 3.")
        final = enhance_input

    # Push result into viewfinder before saving
    if preview_callback is not None and final is not None:
        preview_callback(final)

    suffix = config.OUTPUT_FORMAT
    if config.OUTPUT_NAMING == "timestamp":
        filename = f"openmfc_{int(time.time())}.{suffix}"
    else:
        existing = list(out_path.glob(f"openmfc_*.{suffix}"))
        filename = f"openmfc_{len(existing):04d}.{suffix}"

    final_path = out_path / filename
    final_to_save = np.clip(final, 0, 255).astype(np.uint8)
    cv2.imwrite(str(final_path), final_to_save)
    log_fn(f"[pipeline] Saved → {final_path}")

    if on_done_fn:
        on_done_fn()

    log_fn("[pipeline] Done.")

def run_pipeline_thread(
    image_paths: list[str],
    onnx_session,
    log_fn,
    on_done_fn=None,
    preview_callback=None,
) -> None:
    thread = threading.Thread(
        target=run_pipeline,
        args=(image_paths, onnx_session, log_fn, on_done_fn, preview_callback),
        daemon=True,
    )
    thread.start()


# ─────────────────────────────────────────
#  Live capture
# ─────────────────────────────────────────

def capture_burst(log_fn) -> list[str]:
    cap = cv2.VideoCapture(config.CAPTURE_DEVICE)
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    log_fn(f"[capture] Capturing {config.BURST_COUNT} frames...")
    for i in range(config.BURST_COUNT):
        ret, frame = cap.read()
        if ret:
            path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(path, frame)
            image_paths.append(path)
    cap.release()

    log_fn(f"[capture] Captured {len(image_paths)} frames.")
    return image_paths


# ─────────────────────────────────────────
#  Preview loop
# ─────────────────────────────────────────

def preview_loop(
    is_running_fn,
    on_frame_fn,
    log_fn,
    set_running_fn,
) -> None:
    cap = cv2.VideoCapture(config.CAPTURE_DEVICE, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        log_fn(f"[preview] Could not open camera index {config.CAPTURE_DEVICE}.")
        set_running_fn(False)
        return

    for _ in range(5):
        cap.grab()

    log_fn(f"[preview] Live feed started (camera {config.CAPTURE_DEVICE}).")

    consecutive_failures = 0
    while is_running_fn():
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 10:
                log_fn("[preview] Camera stopped delivering frames.")
                break
            time.sleep(0.05)
            continue
        consecutive_failures = 0
        try:
            on_frame_fn(frame)
        except Exception:
            pass

    cap.release()
    set_running_fn(False)
    log_fn("[preview] Live feed stopped.")