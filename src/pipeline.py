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
    progress_fn=None,
) -> None:
    def _progress(val: int):
        if progress_fn:
            progress_fn(val)

    t0 = time.monotonic()
    _progress(0)
    log_fn(f"[pipeline] Starting with {len(image_paths)} frames...")
    images = load_burst(image_paths)
    if not images:
        log_fn("[pipeline] No valid images loaded. Aborting.")
        _progress(0)
        return

    log_fn(f"[pipeline] Loaded {len(images)} frames — resolution: {images[0].shape[1]}×{images[0].shape[0]}")

    # ── Stage 1: MFNR ──────────────────────────────────────────────
    log_fn(f"[mfnr] ECC threshold: {config.ECC_THRESHOLD} | Burst count: {len(images)}")
    t1 = time.monotonic()

    # Compute per-frame sharpness (Laplacian variance) for logging
    sharpness = []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness.append(lap_var)

    ref_idx = int(np.argmax(sharpness))
    log_fn(f"[mfnr] Reference frame: #{ref_idx} (sharpness {sharpness[ref_idx]:.1f})")
    for i, s in enumerate(sharpness):
        weight = s / max(sum(sharpness), 1e-6)
        marker = " ← ref" if i == ref_idx else ""
        log_fn(f"[mfnr]   frame {i:02d} | sharpness {s:7.1f} | weight {weight:.3f}{marker}")

    mfnr_result = mfnr(images, ecc_threshold=config.ECC_THRESHOLD)
    if mfnr_result is None:
        log_fn("[pipeline] MFNR returned None. Aborting.")
        _progress(0)
        return

    t1e = time.monotonic()
    log_fn(f"[mfnr] Stage 1 done in {t1e - t1:.2f}s")
    _progress(33)

    out_path = Path(config.OUTPUT_PATH)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 2: HDR ───────────────────────────────────────────────
    if config.ENABLE_HDR:
        ev = getattr(config, "HDR_EV_BRACKET", "±1")
        fw = getattr(config, "HDR_FUSION_WEIGHT", 0.5)
        log_fn(f"[hdr] Mode: {getattr(config, 'HDR_MODE', 'auto')} | EV bracket: {ev} | Fusion weight: {fw}")
        t2 = time.monotonic()
        hdr_result = run_hdr(mfnr_result)
        log_fn(f"[hdr] Stage 2 done in {time.monotonic() - t2:.2f}s")
        enhance_input = hdr_result
    else:
        log_fn("[pipeline] HDR disabled, skipping Stage 2.")
        enhance_input = cv2.normalize(
            mfnr_result, None, 0, 255, cv2.NORM_MINMAX
        ).astype("uint8")
    _progress(66)

    # ── Stage 3: Enhance ───────────────────────────────────────────
    if config.ENABLE_ONNX:
        log_fn(f"[enhance] Provider: {getattr(config, 'ONNX_EXECUTION_PROVIDER', 'auto')}")
        t3 = time.monotonic()
        final = run_enhance(enhance_input, session=onnx_session)
        log_fn(f"[enhance] Stage 3 done in {time.monotonic() - t3:.2f}s")
    else:
        log_fn("[pipeline] Enhance disabled, skipping Stage 3.")
        final = enhance_input
    _progress(90)

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
    _progress(100)

    if on_done_fn:
        on_done_fn()

    log_fn(f"[pipeline] Done — total time: {time.monotonic() - t0:.2f}s")

def run_pipeline_thread(
    image_paths: list[str],
    onnx_session,
    log_fn,
    on_done_fn=None,
    preview_callback=None,
    progress_fn=None,
) -> None:
    thread = threading.Thread(
        target=run_pipeline,
        args=(image_paths, onnx_session, log_fn, on_done_fn, preview_callback, progress_fn),
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