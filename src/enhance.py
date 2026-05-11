import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from config import OUTPUT_PATH, ENABLE_ONNX, ONNX_MODEL_PATH, ONNX_EXECUTION_PROVIDER


#Execution provider mapping

_PROVIDER_MAP = {
    "auto":      None,           # let ORT pick best available
    "opencl":    ["OpenCLExecutionProvider",    "CPUExecutionProvider"],
    "directml":  ["DmlExecutionProvider",       "CPUExecutionProvider"],
    "cpu":       ["CPUExecutionProvider"],
}

def _resolve_providers(provider_key: str) -> list[str] | None:
    """Return ORT provider list from config key. None = ORT default (auto)."""
    return _PROVIDER_MAP.get(provider_key.lower(), None)


#Model loading

def load_model(model_path: str) -> ort.InferenceSession | None:
    providers = _resolve_providers(ONNX_EXECUTION_PROVIDER)

    try:
        session = ort.InferenceSession(
            model_path,
            providers=providers if providers else ort.get_available_providers(),
        )
        active = session.get_providers()
        print(f"[enhance] Model loaded | providers: {active}")
        return session
    except Exception as e:
        print(f"[enhance] Failed to load model: {e}")
        return None


#Tensor conversion

def preprocess(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # CHW → NCHW
    return img

def postprocess(tensor: np.ndarray) -> np.ndarray:
    img = np.squeeze(tensor, axis=0)     # NCHW → CHW
    img = np.transpose(img, (1, 2, 0))  # CHW → HWC
    return np.clip(img * 255, 0, 255).astype(np.uint8)


#CV fallback enhancement

def _cv_enhance(image: np.ndarray) -> np.ndarray:
    # Unsharp mask: sharpen = original + (original - blurred) * strength
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # CLAHE on L channel (LAB space) — boosts local contrast without blowing highlights
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return enhanced


#ONNX inference

def _onnx_enhance(image: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Run image through loaded ONNX model."""
    input_name = session.get_inputs()[0].name
    input_tensor = preprocess(image)

    print("[enhance] Running ONNX inference...")
    output = session.run(None, {input_name: input_tensor})
    result = postprocess(output[0])

    print("[enhance] Inference complete")
    return result


#Main entry point

def run_enhance(
    image: np.ndarray,
    session: ort.InferenceSession | None = None,
) -> np.ndarray:
    if ENABLE_ONNX and session is not None:
        return _onnx_enhance(image, session)
    
    print("[enhance] ONNX disabled or session unavailable, skipping.")
    return image  # pass through untouched


#Output

def save_enhance_output(image: np.ndarray, filename: str = "enhanced_output.jpg") -> None:
    output_path = Path(OUTPUT_PATH) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"[enhance] Saved → {output_path}")
