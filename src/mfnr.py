import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from config import OUTPUT_PATH


try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    _RAWPY_AVAILABLE = False


# ── Format routing ─────────────────────────────────────────────────────────────

_RAW_EXTENSIONS  = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf"}
_WEBP_EXTENSIONS = {".webp"}

def _load_single(path: str) -> np.ndarray | None:
    """
    Load one image file to a uint8 BGR ndarray.
    Routes DNG/RAW through rawpy, WebP explicitly through cv2.imdecode
    (avoids the OpenCV IMREAD_COLOR flag skipping transparency on some builds),
    everything else through cv2.imread.
    """
    ext = Path(path).suffix.lower()

    if ext in _RAW_EXTENSIONS:
        if not _RAWPY_AVAILABLE:
            print(f"[warn] rawpy unavailable — skipping RAW file: {path}")
            return None
        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=8,
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[warn] rawpy failed on {path}: {e}")
            return None

    if ext in _WEBP_EXTENSIONS:
        # cv2.imread handles WebP fine on modern builds, but imdecode is safer
        # cross-platform and avoids Unicode path issues on Windows.
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[warn] imdecode returned None for WebP: {path}")
            return img
        except Exception as e:
            print(f"[warn] WebP load failed for {path}: {e}")
            return None

    # Standard path — also uses fromfile to handle Unicode paths on Windows
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] Could not decode: {path}")
        return img
    except Exception as e:
        print(f"[warn] Load failed for {path}: {e}")
        return None


# ── Loading ────────────────────────────────────────────────────────────────────

def load_burst(image_paths: list[str]) -> list[np.ndarray]:
    images = []
    for path in image_paths:
        img = _load_single(path)
        if img is not None:
            images.append(img)
    return images


#Reference selection 

def _sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _pick_reference(images: list[np.ndarray]) -> int:
    scores = [_sharpness(img) for img in images]
    return int(np.argmax(scores))


#ECC alignmen

def _align_to_reference(
    src: np.ndarray,
    ref: np.ndarray,
    motion_model: int = cv2.MOTION_TRANSLATION,
    max_iter: int = 50,
    eps: float = 1e-4,
) -> tuple[np.ndarray, float]:
    scale = 0.25  # align at 1/4 resolution
    ref_small = cv2.resize(ref, None, fx=scale, fy=scale)
    src_small = cv2.resize(src, None, fx=scale, fy=scale)

    ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_gray = cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Warp matrix: 2x3 for translation/euclidean, 3x3 for homography
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

    try:
        ecc_score, warp_matrix = cv2.findTransformECC(
            ref_gray, src_gray, warp_matrix, motion_model, criteria
        )
    except cv2.error:
        # ECC failed to converge — return source unmodified, score = 0
        print(f"[warn] ECC failed to converge. Using raw frame.")
        return src.copy(), 0.0

    h, w = ref.shape[:2]
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(src, warp_matrix, (w, h))
    else:
        warped = cv2.warpAffine(src, warp_matrix, (w, h))

    return warped, float(ecc_score)


#Temporal merge

def _weighted_merge(
    frames: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    total_weight = sum(weights)
    if total_weight == 0:
        # Fallback: naive average
        return np.mean(np.array(frames, dtype=np.float32), axis=0)

    acc = np.zeros_like(frames[0], dtype=np.float32)
    for frame, w in zip(frames, weights):
        acc += frame.astype(np.float32) * w
    return acc / total_weight


#Stage 1: MFNR

def mfnr(
    images: list[np.ndarray],
    motion_model: int = cv2.MOTION_TRANSLATION,
    ecc_threshold: float = 0.3,
) -> np.ndarray | None:
    if not images:
        return None
    if len(images) == 1:
        return images[0].astype(np.float32)

    ref_idx = _pick_reference(images)
    ref = images[ref_idx]
    print(f"[mfnr] Reference frame: {ref_idx} (sharpness={_sharpness(ref):.1f})")

    aligned_frames: list[np.ndarray] = []
    weights: list[float] = []

    for i, frame in enumerate(images):
        if i == ref_idx:
            aligned_frames.append(ref)
            weights.append(1.0)          # reference always gets full weight
            continue

        warped, score = _align_to_reference(frame, ref, motion_model)
        print(f"[mfnr] Frame {i}: ECC score={score:.4f}")

        if score < ecc_threshold:
            print(f"[mfnr] Frame {i} below threshold ({ecc_threshold}), skipping.")
            continue

        aligned_frames.append(warped)
        weights.append(score)

    return _weighted_merge(aligned_frames, weights)


#Output

def save_output(image: np.ndarray, filename: str = "mfnr_output.jpg") -> None:
    """Cast to uint8 and save. Clips float values safely before cast."""
    output_path = Path(OUTPUT_PATH) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clip before cast — float values outside [0, 255] wrap around on direct cast
    out_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), out_uint8)
    print(f"[mfnr] Saved → {output_path}")