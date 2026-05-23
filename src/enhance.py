import cv2
import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from config import OUTPUT_PATH, ENABLE_ONNX, ONNX_MODEL_PATH, ONNX_EXECUTION_PROVIDER
import config

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

    # Auto-convert .pth to .onnx if needed
    if model_path.lower().endswith(".pth"):
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        if not os.path.exists(onnx_path):
            print(f"[enhance] .pth detected — converting to ONNX...")
            try:
                from pytorch2onnx import SRVGGNetCompact
                import torch

                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                ckpt = torch.load(model_path, map_location='cpu')
                key = 'params_ema' if 'params_ema' in ckpt else 'params'
                model.load_state_dict(ckpt[key])
                model.eval()

                x = torch.rand(1, 3, 64, 64)
                with torch.no_grad():
                    torch.onnx.export(
                        model, x, onnx_path,
                        opset_version=11,
                        export_params=True,
                        dynamic_axes={
                            'input': {2: 'height', 3: 'width'},
                            'output': {2: 'height', 3: 'width'}
                        },
                        input_names=['input'],
                        output_names=['output']
                    )
                print(f"[enhance] Converted and saved: {onnx_path}")
            except Exception as e:
                print(f"[enhance] .pth conversion failed: {e}")
                return None
        else:
            print(f"[enhance] Found existing .onnx, skipping conversion: {onnx_path}")
        model_path = onnx_path

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
    """Run image through ONNX model using tiling to handle large images."""
    input_name = session.get_inputs()[0].name
    tile_size = 256
    overlap = 16
    scale = 4

    max_h = config.ENHANCE_MAX_HEIGHT
    h, w = image.shape[:2]
    if h > max_h:
        scale_down = max_h / h
        new_w = int(w * scale_down)
        image = cv2.resize(image, (new_w, max_h), interpolation=cv2.INTER_AREA)
        print(f"[enhance] Downscaled input to {new_w}x{max_h} before inference")

    input_name = session.get_inputs()[0].name

    h, w = image.shape[:2]
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 3), dtype=np.float32)

    print(f"[enhance] Tiled inference {w}x{h} → {out_w}x{out_h}")

    y = 0
    while y < h:
        x = 0
        while x < w:
            # crop tile with overlap
            x1 = max(x - overlap, 0)
            y1 = max(y - overlap, 0)
            x2 = min(x + tile_size + overlap, w)
            y2 = min(y + tile_size + overlap, h)

            tile = image[y1:y2, x1:x2]
            tensor = preprocess(tile)
            out = session.run(None, {input_name: tensor})[0]
            out_tile = postprocess(out)

            # output coords
            ox1, oy1 = x1 * scale, y1 * scale
            ox2, oy2 = x2 * scale, y2 * scale

            output[oy1:oy2, ox1:ox2] += out_tile.astype(np.float32)
            weight[oy1:oy2, ox1:ox2] += 1.0

            x += tile_size
        y += tile_size

    output = np.clip(output / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)
    print("[enhance] Inference complete")
    return output


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
