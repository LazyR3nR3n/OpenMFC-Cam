# OpenMFC-Cam
**Open Multi-Frame Compounding Camera**

![OSI Approved](https://img.shields.io/badge/license-GPLv3-brightgreen?logo=open-source-initiative)
![Python](https://img.shields.io/badge/Python-3.12+-yellow)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![GitHub release](https://img.shields.io/github/v/release/LazyR3nR3n/OpenMFC-Cam?include_prereleases)

A standalone desktop computational MFNR photography tool that takes a burst of frames — either captured live from a webcam/USB camera or loaded manually — and produces a single cleaned, enhanced output image using physics-based noise reduction combined with optional ML enhancement.

---
## Table of Contents
- [How It Works](#how-it-works)
- [UI Layout](#ui-layout)
- [ONNX Model Swap](#onnx-model-swap)
- [Running from Source](#running-from-source)
- [Download](#download)
- [What's New in Beta 0.01.1](#whats-new-in-beta-0011)
- [Beta Limitations](#beta-limitations)
- [System Requirements](#system-requirements)
- [Support](#support)
- [License](#license)





---

## How It Works

**MFNRC (Multi-Frame Noise Reduction Compounding)** is the core algorithm:

1. **Burst Capture** — captures multiple frames from a live camera or accepts manually loaded JPEGs (Note: The live camera field with live Camera input are untested as of current release)
2. **ECC Alignment** — aligns frames using Enhanced Correlation Coefficient to reject motion-blurred or misaligned frames
3. **MFNR** — edge-aware weighted averaging across aligned frames to suppress noise while preserving detail
4. **HDR Fusion** — Mertens exposure fusion on the denoised output for highlight/shadow recovery (Auto / Enable / Disable)
5. **ONNX Enhancement** — optional ML upscaling/enhancement via ONNX Runtime (default: Real-ESRGAN (realesr-general-x4v3) ; and other alternatives.)

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
| Real-ESRGAN (realesr-general-x4v3) | Default — fast upscaling | Recommended for general use |
| SwinIR | High-quality restoration | Slower, better detail |
| NAFNet | Deblurring + denoising | ALternative backend |

Set your `.onnx` file path in Settings and select the execution provider (Auto / OpenCL / DirectML / CPU).

---

## Running from Source

**Requirements:** Python 3.12+

```bash
git clone https://github.com/LazyR3nR3n/OpenMFC-Cam.git
cd OpenMFC-Cam
python -m venv .venv
.venv\Scripts\activate

# Required
pip install opencv-python pyqt6 rawpy numpy onnxruntime Pillow requests

# Optional — only needed for .pth model auto-conversion
pip install torch
```

---

## Download

Pre-built Windows executable available under [Releases](https://github.com/LazyR3nR3n/OpenMFC-Cam/releases).

No Python installation required.

---
## What's New in Beta 0.01.1

* Auto-updater — OpenMFC now checks for new releases on startup. A dialog appears when an update is available, letting you download and install it in one click.
* PowerShell startup fix — The black console window that flashed on startup is now gone.
* `.pth` auto-conversion — Power users can now load `.pth` model files directly; OpenMFC converts them to `.onnx` automatically on first load.(Currenltly .pth auto-conversion currently supports Real-ESRGAN architecture only)
* ONNX model path fix — Model path now resolves correctly after installation (no more "model not found" on startup).


IMPORTANT NOTE: MFNR resolution cap is adjustable in settings — default set in 1080p for performance. You can change it between 480–4320 (4K). Lower = faster, less detail. Higher = slower, more detail. (I decided this since I noticed that it is eating resources like it is on a feast, on my laptop and I got an old one so yeah, BUT if you have a better Laptop or PC this shouldn't be an issue at all since the default ML model are ALREADY lightweight. I haven't try any other model though so feel free to give feedback how it goes, feedback is much so appreciated. -LazyR3nR3n)


---

## Beta Limitations

---
⚠️ Known issues in Beta 0.01.1:

* Manual camera detection not yet implemented (planned for v0.02.0).
* `.pth` auto-conversion only supports SRVGGNetCompact architecture.
* Live camera input via USB is untested — feedback welcome.


 
---


## System Requirements

| | |
|---|---|
| **OS** | Windows 10 / 11 (64-bit) |
| **RAM** | 8GB minimum, 16GB recommended(or higher if using a heavier onnx model)|
| **Camera** | Any USB or built-in webcam (Alternatively, Use Open Camera Burst capture mode and add it through input)|
| **GPU** | Use for faster results— enables OpenCL/DirectML ONNX acceleration |

---

## Support

If OpenMFC has been useful to you, consider buying me a coffee!

☕ [Ko-fi — LazyR3nR3n](https://ko-fi.com/lazyr3nr3n)

---

## License

GPLv3 — see [LICENSE](LICENSE) for full terms.

```
OpenMFC-Cam Copyright (C) 2026 LazyR3nR3n
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE for details.
```

