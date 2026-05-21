
import os



# Burst settings
BURST_COUNT: int = 20

# Output
OUTPUT_PATH: str = os.path.join(os.path.expanduser("~"), "Pictures", "OpenMFC")
OUTPUT_FORMAT: str = "jpg"          # "jpg" | "png"
OUTPUT_NAMING: str = "timestamp"    # "timestamp" | "sequential"

# Processing — MFNR
ECC_THRESHOLD: float = 0.3          # frames below this ECC score are dropped

# Processing — HDR
ENABLE_HDR: bool = False
HDR_MODE: str = "auto"              # "auto" | "enable" | "disable"
HDR_EV_BRACKET: str = "±1"         # "±1" | "±2" | "±3" | "Custom"
HDR_FUSION_WEIGHT: float = 0.5     # 0.0 = bias shadows, 1.0 = bias highlights

# Processing — ONNX Enhancement
ENABLE_ONNX: bool = False           # keep False until model is present
ONNX_MODEL_PATH: str = "assets/models/realesrgan_lite.onnx"
ONNX_EXECUTION_PROVIDER: str = "auto"  # "auto" | "opencl" | "directml" | "cpu"

# Capture device
CAPTURE_DEVICE: int = 0             # updated at runtime by devices.enumerate_cameras()
CAPTURE_WIDTH: int = 1280
CAPTURE_HEIGHT: int = 720
CAPTURE_FPS: int = 30




#by LazyR3nR3n