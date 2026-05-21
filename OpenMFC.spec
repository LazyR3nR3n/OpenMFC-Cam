# ─────────────────────────────────────────────────────────────────────────────
#  OpenMFC Beta 0.01.0 — PyInstaller Spec
#  Build: pyinstaller OpenMFC.spec
#  Target: Windows x64 (one-folder build, see notes below)
# ─────────────────────────────────────────────────────────────────────────────
#
#  BEFORE BUILDING:
#  1. Activate your venv
#  2. pip install pyinstaller
#  3. Place this file in the project root (same folder as main.py)
#  4. Run: pyinstaller OpenMFC.spec
#  5. Output is in dist/OpenMFC/
#
#  ONE-FOLDER vs ONE-FILE:
#  We use onedir (not onefile) intentionally. Onefile unpacks to a temp
#  directory on every launch which causes a 2-5s cold-start spike — exactly
#  the kind of thing we already fixed in main.py. Onedir starts instantly.
#  Ship the whole dist/OpenMFC/ folder as a .zip on GitHub releases.
#
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Source files ──────────────────────────────────────────────────────────────

source_files = [
    'main.py',
    'ui.py',
    'config.py',
    'pipeline.py',
    'mfnr.py',
    'hdr.py',
    'enhance.py',
    'devices.py',
]

# ── Hidden imports ────────────────────────────────────────────────────────────
#
#  PyInstaller misses these because they are:
#  - Inside try/except ImportError blocks (rawpy)
#  - Loaded dynamically at runtime by onnxruntime
#  - PyQt6 backend plugins not auto-collected

hidden_imports = [
    # rawpy — optional import, PyInstaller won't trace through try/except
    'rawpy',
    'rawpy._rawpy',

    # onnxruntime providers — loaded via string at runtime
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',

    # PyQt6 — ensure platform plugin and core modules are included
    'PyQt6.QtWidgets',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.sip',

    # PIL/Pillow — used in ensure_icon()
    'PIL.Image',
    'PIL.PngImagePlugin',
    'PIL.JpegImagePlugin',

    # numpy internals sometimes missed
    'numpy.core._methods',
    'numpy.lib.format',
]

# ── Data files ────────────────────────────────────────────────────────────────
#
#  (source, dest_folder_inside_bundle)
#  assets/ is the icon folder — add models/ subfolder here too when ready

datas = [
    # Project assets (icon png + generated ico)
    ('assets', 'assets'),
]

# Collect onnxruntime data files (providers, schemas, etc.)
datas += collect_data_files('onnxruntime')

# ── Binaries ──────────────────────────────────────────────────────────────────
#
#  Native .dll/.so files PyInstaller won't auto-detect.

binaries = []

# onnxruntime native libs
binaries += collect_dynamic_libs('onnxruntime')

# rawpy native libs (libraw)
binaries += collect_dynamic_libs('rawpy')

# ── Analysis ──────────────────────────────────────────────────────────────────

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages we definitely don't use
        'matplotlib',
        'scipy',
        'pandas',
        'tkinter',
        'wx',
        'PyQt5',
        'PySide2',
        'PySide6',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ── PYZ ───────────────────────────────────────────────────────────────────────

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── EXE ───────────────────────────────────────────────────────────────────────

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # onedir: binaries go in COLLECT, not embedded
    name='OpenMFC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                    # compress with UPX if available — reduces size
    console=False,               # no console window (equiv to --noconsole)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows icon
    icon=os.path.join('assets', 'icon.ico'),
)

# ── COLLECT ───────────────────────────────────────────────────────────────────
#
#  Assembles the final dist/OpenMFC/ folder

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Don't UPX these — UPX breaks them on some Windows configs
        'vcruntime*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',
        'Qt6*.dll',
    ],
    name='OpenMFC',
)
