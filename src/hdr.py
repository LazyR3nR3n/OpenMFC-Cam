import cv2
import numpy as np
from pathlib import Path
from config import OUTPUT_PATH


#Synthetic exposure generation

def _generate_exposure_variants(
    image: np.ndarray,
    stops: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
) -> list[np.ndarray]:
    variants = []
    for stop in stops:
        factor = 2.0 ** stop
        shifted = np.clip(image * factor, 0, 255).astype(np.uint8)
        variants.append(shifted)
    return variants


#Mertens fusion

def merge_mertens(frames: list[np.ndarray]) -> np.ndarray:
    """
    Apply Mertens exposure fusion on a list of uint8 frames.
    Mertens outputs float32 in [0, 1] — converted back to uint8 before return.
    """
    merge = cv2.createMergeMertens()
    fused = merge.process(frames)
    return np.clip(fused * 255, 0, 255).astype(np.uint8)


#Stage 2: HDR

def run_hdr(
    mfnr_output: np.ndarray,
    stops: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
) -> np.ndarray:
    if mfnr_output is None:
        return None

    print(f"[hdr] Generating {len(stops)} synthetic exposure variants...")
    variants = _generate_exposure_variants(mfnr_output, stops)

    print("[hdr] Running Mertens fusion...")
    result = merge_mertens(variants)

    print("[hdr] Fusion complete")
    return result


#Output

def save_hdr_output(image: np.ndarray, filename: str = "hdr_output.jpg") -> None:
    output_path = Path(OUTPUT_PATH) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"[hdr] Saved → {output_path}")

