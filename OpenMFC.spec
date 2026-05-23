# ─────────────────────────────────────────────────────────────────────────────
#  OpenMFC Beta 0.01.1 — PyInstaller Spec
#  Build: pyinstaller OpenMFC.spec
#  Target: Windows x64 (one-folder build)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Hidden imports ────────────────────────────────────────────────────────────

hidden_imports = [
    # rawpy — optional import, PyInstaller won't trace through try/except
    'rawpy',
    'rawpy._rawpy',

    # onnxruntime providers — loaded via string at runtime
    'onnxruntime.capi._pybind_state',
    'onnxruntime.capi.onnxruntime_pybind11_state',

    # PyQt6
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

    # requests — used by updater.py for GitHub releases API
    'requests',
    'requests.adapters',
    'requests.packages.urllib3',
]

# ── Data files ────────────────────────────────────────────────────────────────

datas = [
    ('assets', 'assets'),
    ('assets/models', 'models'),
]

datas += collect_data_files('onnxruntime')

# ── Binaries ──────────────────────────────────────────────────────────────────

binaries = []
binaries += collect_dynamic_libs('onnxruntime')
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
        'torch',
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
    exclude_binaries=True,
    name='OpenMFC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join('assets', 'icon.ico'),
)

# ── COLLECT ───────────────────────────────────────────────────────────────────

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime*.dll',
        'msvcp*.dll',
        'api-ms-*.dll',
        'Qt6*.dll',
    ],
    name='OpenMFC',
)
