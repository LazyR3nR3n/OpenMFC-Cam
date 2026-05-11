import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from config import OUTPUT_PATH


#Loading

def load_burst(image_paths: list[str]) -> list[np.ndarray]:
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"[warn] Could not read: {path}")
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
    max_iter: int = 200,
    eps: float = 1e-4,
) -> tuple[np.ndarray, float]:
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)

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