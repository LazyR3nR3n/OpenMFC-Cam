# OpenMFC-Cam
**Open Multi-Frame Compounding Camera**

A desktop computational photography pipeline that takes a burst of frames — either captured live from a webcam/USB camera or loaded manually — and produces a single cleaned, enhanced output image using physics-based noise reduction combined with optional ML enhancement.

---

## How It Works

**MFNRC (Multi-Frame Noise Reduction Compounding)** is the core algorithm:

1. **Burst Capture** — captures multiple frames from a live camera or accepts manually loaded JPEGs
2. **ECC Alignment** — aligns frames using Enhanced Correlation Coefficient to reject motion-blurred or misaligned frames
3. **MFNR** — edge-aware weighted averaging across aligned frames to suppress noise while preserving detail
4. **HDR Fusion** — Mertens exposure fusion on the denoised output for highlight/shadow recovery (Auto / Enable / Disable)
5. **ONNX Enhancement** — optional ML upscaling/enhancement via ONNX Runtime (default: Real-ESRGAN lite; and other alternatives.)

---

## UI Layout

Three-panel OBS-style interface:

| Panel | Contents |
|-------|----------|
| **Left** | Project Folder (output gallery + thumbnails) · Input (manual burst file loader) |
| **Center** | Histogram · Viewfinder · Log · Action Buttons |
| **Right** | OPTIONS — MFNR, HDR, Enhance, Burst Count, Camera Preview, Viewfinder Ratio, View Output |

---

## ONNX Model Swap

The enhance stage is hot-swappable via Settings → ONNX Model Path.

| Model | Purpose | Notes |
|-------|---------|-------|
| Real-ESRGAN lite | Default — fast upscaling | Recommended for general use |
| SwinIR | High-quality restoration | Slower, better detail |
| NAFNet | Deblurring + denoising | ALternative backend |

Set your `.onnx` file path in Settings and select the execution provider (Auto / OpenCL / DirectML / CPU).

---

## Running from Source

**Requirements:** Python 3.10+

```bash
git clone https://github.com/LazyR3nR3n/OpenMFC-Cam.git
cd OpenMFC-Cam
python -m venv .venv
.venv\Scripts\activate
pip install opencv-python dearpygui numpy onnxruntime Pillow
python src/main.py
```

---

## Download

Pre-built Windows executable available under [Releases](https://github.com/LazyR3nR3n/OpenMFC-Cam/releases).

No Python installation required.

---

## License

GPLv3 — see [LICENSE](LICENSE) for full terms.

```
OpenMFC-Cam Copyright (C) 2026 LazyR3nR3n
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE for details.
```

