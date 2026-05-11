import cv2
import threading
import time
import config
import numpy as np
import dearpygui.dearpygui as dpg
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

from devices import get_index_from_label
from pipeline import (
    run_pipeline_thread,
    capture_burst,
    preview_loop,
)


# ─────────────────────────────────────────
#  Layout constants
# ─────────────────────────────────────────

LEFT_W   = 224
RIGHT_W  = 218          # slightly wider so cards don't clip
TOPBAR_H = 34
HIST_H   = 60
LOG_H    = 124          # bumped from 112 → gives buttons room to breathe
GAP      = 2

# Right-panel card heights — no longer used (collapsing headers are content-sized)


# ─────────────────────────────────────────
#  Geometry helper
# ─────────────────────────────────────────

def _geometry(vp_w: int, vp_h: int) -> dict:
    body_h   = max(300, vp_h - TOPBAR_H - 8)
    center_w = max(300, vp_w - LEFT_W - RIGHT_W - 12)

    # OVERHEAD accounts for all real pixel costs DPG consumes inside center_col:
    #   center_col window padding (12), hist_panel border+padding (~10),
    #   vf_panel border+padding (~10), log_panel border+padding (~10), GAP*3 spacers
    OVERHEAD     = HIST_H + LOG_H + GAP * 3 + 42
    avail_for_vf = max(80, body_h - OVERHEAD)
    vf_h         = _vf_target_h(center_w, avail_for_vf)

    left_top = max(120, body_h // 2 - GAP)
    left_bot = max(120, body_h - left_top - GAP)
    return dict(body_h=body_h, center_w=center_w, vf_h=vf_h,
                left_top=left_top, left_bot=left_bot)


def _vf_target_h(center_w: int, max_h: int) -> int:
    ratios = {"16:9": 9/16, "4:3": 3/4, "1:1": 1.0}
    if _vf_aspect in ratios:
        return max(80, min(max_h, int((center_w - 18) * ratios[_vf_aspect])))
    return max(80, max_h)   # Free


# ─────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────

_onnx_session     = None
_camera_labels:   list[str] = []
_input_paths:     list[str] = []
_preview_running  = False
_preview_thread   = None
_live_view_active = False

# Thread-safe frame buffer
_pending_frame:      np.ndarray | None = None
_pending_frame_lock: threading.Lock    = threading.Lock()
_frame_dirty:        bool              = False
_last_flush_time:    float             = 0.0
_FLUSH_INTERVAL:     float             = 1 / 30

# "Static" image shown in viewfinder when camera is off
# Set by: pipeline completion, thumbnail click, open-output-file
_static_frame:      np.ndarray | None = None
_static_dirty:      bool              = False

_vf_aspect: str = "Free"

# Track thumbnail tags so we can wire click callbacks
_thumb_tags: list[str] = []


# ─────────────────────────────────────────
#  Viewport resize
# ─────────────────────────────────────────

def _on_viewport_resize(sender, app_data) -> None:
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    g    = _geometry(vp_w, vp_h)
    try:
        dpg.set_item_width ("main_window",     vp_w)
        dpg.set_item_height("main_window",     vp_h)
        dpg.set_item_height("left_top",        g["left_top"])
        dpg.set_item_height("left_bot",        g["left_bot"])
        dpg.set_item_height("project_gallery", max(40, g["left_top"] - 94))
        dpg.set_item_height("input_file_box",  max(40, g["left_bot"] - 74))
        dpg.set_item_width ("center_col",      g["center_w"])
        dpg.set_item_height("center_col",      g["body_h"])
        dpg.set_item_width ("hist_panel",      g["center_w"] - 2)
        dpg.set_item_width ("hist_draw",       g["center_w"] - 20)
        dpg.configure_item ("hist_draw",       width=g["center_w"] - 20)
        dpg.set_item_height("vf_panel",        g["vf_h"])
        dpg.set_item_width ("vf_panel",        g["center_w"] - 2)
        dpg.set_item_width ("vf_image",        g["center_w"] - 18)
        dpg.set_item_height("vf_image",        g["vf_h"] - 10)
        dpg.set_item_width ("log_panel",       g["center_w"] - 152)
        dpg.set_item_height("log_panel",       LOG_H)   # pin log height on resize
        dpg.set_item_height("btn_panel",       LOG_H)   # pin btn height on resize
        # right_panel height is not clamped — scrolls if cards overflow
        # title spacer
        BTN_END = 184
        TITLE_W = 38 * 7
        sw = max(4, LEFT_W + g["center_w"] // 2 - BTN_END - TITLE_W // 2)
        dpg.set_item_width("title_spacer", sw)
    except Exception:
        pass


# ─────────────────────────────────────────
#  Theme
# ─────────────────────────────────────────

def _apply_theme(mode: str = "dark") -> None:
    if mode == "auto":
        import platform
        if platform.system() == "Windows":
            try:
                import winreg
                k = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
                mode = "light" if v == 1 else "dark"
            except Exception:
                mode = "dark"
        else:
            mode = "dark"

    with dpg.theme() as app_theme:
        with dpg.theme_component(dpg.mvAll):
            if mode == "dark":
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (22,  22,  22))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         (32,  32,  32))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         (18,  18,  18))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (50,  50,  50))
                dpg.add_theme_color(dpg.mvThemeCol_Button,          (52,  52,  52))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   (70,  70,  70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    (95,  95,  95))
                dpg.add_theme_color(dpg.mvThemeCol_Header,          (50,  50,  50))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   (68,  68,  68))
                dpg.add_theme_color(dpg.mvThemeCol_Text,            (228, 226, 220))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,    (110, 110, 110))
                dpg.add_theme_color(dpg.mvThemeCol_Border,          (52,  52,  52))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     (18,  18,  18))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   (55,  55,  55))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         (16,  16,  16))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   (22,  22,  22))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         (28,  28,  28))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,       (200, 200, 200))
            else:
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (238, 238, 238))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         (222, 222, 222))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (185, 185, 185))
                dpg.add_theme_color(dpg.mvThemeCol_Button,          (208, 208, 208))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   (188, 188, 188))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    (168, 168, 168))
                dpg.add_theme_color(dpg.mvThemeCol_Text,            (20,  20,  20))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,    (130, 130, 130))
                dpg.add_theme_color(dpg.mvThemeCol_Border,          (178, 178, 178))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         (232, 232, 232))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,       (60,  60,  60))

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding,    4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6, 3)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    6, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize,  1)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize,    8)

    dpg.bind_theme(app_theme)
    config.THEME = mode


# ─────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────

def log(message: str) -> None:
    try:
        cur = dpg.get_value("log_box")
        ts  = time.strftime("%H:%M:%S")
        dpg.set_value("log_box", cur + f"[{ts}] {message}\n")
    except Exception:
        print(message)


# ─────────────────────────────────────────
#  Textures
# ─────────────────────────────────────────

_VF_W, _VF_H = 640, 360


def _init_viewfinder_texture() -> None:
    blank = np.zeros((_VF_H, _VF_W, 4), dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_dynamic_texture(_VF_W, _VF_H, blank.ravel(), tag="vf_texture")


def _push_frame_to_texture(frame: np.ndarray) -> None:
    """Push a BGR frame directly into the DPG texture. MUST be called from main thread."""
    r    = cv2.resize(frame, (_VF_W, _VF_H))
    rgba = cv2.cvtColor(r, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    dpg.set_value("vf_texture", rgba.ravel())
    _update_histogram(frame)


def _frame_to_texture(frame: np.ndarray) -> None:
    """Called from preview background thread — writes to shared buffer only."""
    global _pending_frame, _frame_dirty
    with _pending_frame_lock:
        _pending_frame = frame.copy()
        _frame_dirty   = True


def _flush_pending_frame() -> None:
    """Called from main-thread render loop — flushes live camera frames."""
    global _pending_frame, _frame_dirty, _last_flush_time, _static_dirty, _static_frame
    now = time.monotonic()

    # Static image takes priority when camera is off
    if _static_dirty and _static_frame is not None and not _preview_running:
        _push_frame_to_texture(_static_frame)
        _static_dirty = False
        return

    if now - _last_flush_time < _FLUSH_INTERVAL:
        return
    with _pending_frame_lock:
        if not _frame_dirty or _pending_frame is None:
            return
        frame        = _pending_frame
        _frame_dirty = False
    _last_flush_time = now
    _push_frame_to_texture(frame)


def show_image_in_viewfinder(img: np.ndarray) -> None:
    """
    Public entry point — call from pipeline.py or anywhere to display
    a BGR image in the viewfinder. Safe to call from any thread.
    """
    global _static_frame, _static_dirty
    _static_frame = img.copy()
    _static_dirty = True
    _update_histogram(img)


def _init_thumbnail_texture(tag: str, img: np.ndarray, w: int = 96, h: int = 72) -> None:
    r    = cv2.resize(img, (w, h))
    rgba = cv2.cvtColor(r, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    with dpg.texture_registry():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, rgba.ravel())
        else:
            dpg.add_dynamic_texture(w, h, rgba.ravel(), tag=tag)


# ─────────────────────────────────────────
#  Histogram
# ─────────────────────────────────────────

def _update_histogram(frame: np.ndarray) -> None:
    if not dpg.does_item_exist("hist_draw"):
        return
    try:
        sz = dpg.get_item_configuration("hist_draw")
        dw = sz.get("width",  600)
        dh = sz.get("height", 32)
    except Exception:
        dw, dh = 600, 32

    dpg.delete_item("hist_draw", children_only=True)
    dpg.draw_rectangle([0, 0], [dw, dh],
        fill=(18, 18, 18, 255), color=(0, 0, 0, 0), parent="hist_draw")
    for frac in (0.25, 0.5, 0.75):
        x = int(dw * frac)
        dpg.draw_line([x, 0], [x, dh],
            color=(255, 255, 255, 20), thickness=1, parent="hist_draw")

    bins  = 64
    step  = dw / bins
    chans = [(0, (90,90,255,190)), (1, (80,210,80,190)), (2, (255,80,80,190))]
    for ch, col in chans:
        hist = cv2.calcHist([frame], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist, 0, dh - 2, cv2.NORM_MINMAX)
        flat = hist.flatten()
        for i in range(bins - 1):
            dpg.draw_line(
                [i * step,       dh - flat[i]],
                [(i + 1) * step, dh - flat[i + 1]],
                color=col, thickness=1, parent="hist_draw")


# ─────────────────────────────────────────
#  Project gallery
# ─────────────────────────────────────────

def _on_thumbnail_click(sender, app_data, user_data) -> None:
    """Load the clicked output image into the viewfinder."""
    path = user_data
    img  = cv2.imread(str(path))
    if img is not None:
        show_image_in_viewfinder(img)
        log(f"[preview] Loaded: {Path(path).name}")
    else:
        log(f"[preview] Could not read: {path}")


def _refresh_project_folder() -> None:
    global _thumb_tags
    folder = Path(config.OUTPUT_PATH)
    if not folder.exists():
        return
    dpg.delete_item("project_gallery", children_only=True)
    _thumb_tags = []

    exts   = {".jpg", ".jpeg", ".png"}
    images = sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in exts],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )[:20]

    if not images:
        dpg.add_text("No outputs yet.", color=(90, 90, 90), parent="project_gallery")
        return

    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        if img is None:
            continue
        tag = f"thumb_tex_{i}"
        _thumb_tags.append(tag)
        _init_thumbnail_texture(tag, img)

        with dpg.group(parent="project_gallery", horizontal=False):
            # Clickable image button — user_data carries the file path
            dpg.add_image_button(
                tag,
                width=96, height=72,
                tag=f"thumb_btn_{i}",
                callback=_on_thumbnail_click,
                user_data=str(p),
            )
            dpg.add_text(p.name[:14], color=(130, 130, 130))
        dpg.add_spacer(height=4, parent="project_gallery")


# ─────────────────────────────────────────
#  Open output file into viewfinder
# ─────────────────────────────────────────

def _on_open_output_file(sender, app_data) -> None:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    f = filedialog.askopenfilename(
        title="Open Output Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tiff *.tif"), ("All files", "*.*")],
        initialdir=config.OUTPUT_PATH,
    )
    root.destroy()
    if f:
        img = cv2.imread(f)
        if img is not None:
            show_image_in_viewfinder(img)
            log(f"[preview] Opened: {Path(f).name}")
        else:
            log(f"[preview] Could not read file: {f}")


# ─────────────────────────────────────────
#  Camera preview
# ─────────────────────────────────────────

def _update_capture_btn() -> None:
    try:
        dpg.configure_item("live_capture_btn", enabled=_live_view_active)
    except Exception:
        pass


def _start_preview() -> None:
    global _preview_running, _live_view_active
    _preview_running  = True
    _live_view_active = True
    _update_capture_btn()

    def _set_running(val):
        global _preview_running, _live_view_active
        _preview_running  = val
        if not val:
            _live_view_active = False
            _update_capture_btn()
            try:
                dpg.set_item_label("connect_camera_btn", "Connect Camera")
                dpg.configure_item("connect_camera_btn", enabled=True)
            except Exception:
                pass

    threading.Thread(
        target=preview_loop,
        args=(lambda: _preview_running, _frame_to_texture, log, _set_running),
        daemon=True,
    ).start()
    log("[camera] Live preview started.")
    try:
        dpg.set_item_label("connect_camera_btn", "Disconnect")
        dpg.configure_item("connect_camera_btn", enabled=True)
    except Exception:
        pass


def _clear_viewfinder() -> None:
    global _pending_frame, _frame_dirty
    time.sleep(0.08)
    with _pending_frame_lock:
        _pending_frame = None
        _frame_dirty   = False
    try:
        blank = np.zeros((_VF_H, _VF_W, 4), dtype=np.float32)
        dpg.set_value("vf_texture", blank.ravel())
    except Exception:
        pass


def _stop_preview() -> None:
    global _preview_running, _live_view_active
    _preview_running  = False
    _live_view_active = False
    _update_capture_btn()
    log("[camera] Live preview stopped.")
    threading.Thread(target=_clear_viewfinder, daemon=True).start()


def _on_connect_camera(sender, app_data) -> None:
    if _preview_running:
        _stop_preview()
        dpg.set_item_label("connect_camera_btn", "Connect Camera")
    else:
        dpg.set_item_label("connect_camera_btn", "Connecting…")
        dpg.configure_item("connect_camera_btn", enabled=False)
        threading.Thread(target=_start_preview, daemon=True).start()


# ─────────────────────────────────────────
#  Pipeline callbacks
# ─────────────────────────────────────────

def _pipeline_done_callback() -> None:
    """Called by pipeline when processing finishes — refresh gallery."""
    _refresh_project_folder()
    # pipeline.py should call show_image_in_viewfinder(result_img) directly
    # before calling this, so the viewfinder updates with the output.


def _on_live_capture_start(sender, app_data) -> None:
    if not _live_view_active:
        log("[capture] No live view — connect camera first.")
        return
    paths = capture_burst(log)
    run_pipeline_thread(paths, _onnx_session, log,
                        _pipeline_done_callback, show_image_in_viewfinder)


def _on_mfnr_input_start(sender, app_data) -> None:
    if not _input_paths:
        log("[input] No images loaded.")
        return
    paths = _input_paths[: config.BURST_COUNT]
    log(f"[input] Running pipeline on {len(paths)} image(s)…")
    run_pipeline_thread(paths, _onnx_session, log,
                        _pipeline_done_callback, show_image_in_viewfinder)


def _on_input_browse(sender, app_data) -> None:
    global _input_paths
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    files = filedialog.askopenfilenames(
        title="Select Burst Images",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.tiff *.tif"), ("All files", "*.*")],
    )
    root.destroy()
    if files:
        _input_paths = list(files)
        dpg.set_value("input_label", f"{len(_input_paths)} file(s) loaded")
        log(f"[input] Loaded {len(_input_paths)} file(s).")
        img = cv2.imread(_input_paths[0])
        if img is not None:
            show_image_in_viewfinder(img)   # preview first input image immediately
    else:
        log("[input] No files selected.")


# ─────────────────────────────────────────
#  OPTIONS callbacks
# ─────────────────────────────────────────

def _on_ecc_changed(s, a)   -> None: config.ECC_THRESHOLD = round(a, 4)
def _on_burst_changed(s, a) -> None: config.BURST_COUNT   = max(1, int(a))

def _on_hdr_radio(s, a) -> None:
    config.HDR_MODE   = a.lower()
    config.ENABLE_HDR = a.lower() == "enable"
    log(f"[options] HDR: {a}")

def _on_enhance_radio(s, a) -> None:
    config.ENABLE_ONNX = a.lower() == "enable"
    log(f"[options] Enhance: {a}")


def _on_aspect_radio(s, a) -> None:
    global _vf_aspect
    _vf_aspect = a
    try: dpg.set_value("ar_radio_b", "")
    except Exception: pass
    _apply_vf_aspect()
    log(f"[viewfinder] Aspect ratio: {a}")


def _on_aspect_radio_b(s, a) -> None:
    global _vf_aspect
    if not a:
        return
    _vf_aspect = a
    try: dpg.set_value("ar_radio", "")
    except Exception: pass
    _apply_vf_aspect()
    log(f"[viewfinder] Aspect ratio: {a}")


def _apply_vf_aspect() -> None:
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    g    = _geometry(vp_w, vp_h)
    try:
        dpg.set_item_height("vf_panel", g["vf_h"])
        dpg.set_item_height("vf_image", g["vf_h"] - 10)
    except Exception:
        pass


# ─────────────────────────────────────────
#  Settings / Help popups
# ─────────────────────────────────────────

def _on_settings_open(s, a)   -> None: dpg.configure_item("settings_popup", show=True)
def _on_settings_cancel(s, a) -> None: dpg.configure_item("settings_popup", show=False)
def _on_help_open(s, a)       -> None: dpg.configure_item("help_popup",     show=True)

def _on_output_browse(s, a) -> None:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    f = filedialog.askdirectory(title="Select Output Folder"); root.destroy()
    if f: dpg.set_value("set_output_path", f)

def _on_onnx_browse(s, a) -> None:
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    f = filedialog.askopenfilename(title="Select ONNX Model",
        filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")]); root.destroy()
    if f: dpg.set_value("set_onnx_path", f)

def _on_settings_save(s, a) -> None:
    config.OUTPUT_PATH             = dpg.get_value("set_output_path")
    config.OUTPUT_FORMAT           = dpg.get_value("set_output_format").lower()
    config.OUTPUT_NAMING           = dpg.get_value("set_output_naming").lower()
    config.ONNX_MODEL_PATH         = dpg.get_value("set_onnx_path")
    config.ONNX_EXECUTION_PROVIDER = dpg.get_value("set_onnx_provider").lower()
    config.CAPTURE_DEVICE          = get_index_from_label(dpg.get_value("set_camera_combo"))
    config.CAPTURE_WIDTH           = int(dpg.get_value("set_cap_width"))
    config.CAPTURE_HEIGHT          = int(dpg.get_value("set_cap_height"))
    config.CAPTURE_FPS             = int(dpg.get_value("set_cap_fps"))
    _apply_theme(dpg.get_value("set_theme").lower())
    dpg.configure_item("settings_popup", show=False)
    log("[settings] Settings saved.")
    _refresh_project_folder()


def _build_settings_popup() -> None:
    with dpg.window(label="Settings", tag="settings_popup",
                    modal=True, show=False, width=490, height=530,
                    no_resize=True, pos=(160, 60)):
        dpg.add_text("OUTPUT", color=(160, 160, 160))
        dpg.add_separator(); dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="set_output_path", default_value=config.OUTPUT_PATH,
                               width=340, hint="Output folder path")
            dpg.add_button(label="Browse", callback=_on_output_browse)
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_text("Format:")
            dpg.add_combo(items=["JPG","PNG"], tag="set_output_format",
                          default_value=config.OUTPUT_FORMAT.upper(), width=110)
            dpg.add_spacer(width=14)
            dpg.add_text("Naming:")
            dpg.add_combo(items=["Timestamp","Sequential"], tag="set_output_naming",
                          default_value=config.OUTPUT_NAMING.capitalize(), width=130)
        dpg.add_spacer(height=12)
        dpg.add_text("ONNX MODEL", color=(160, 160, 160))
        dpg.add_separator(); dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="set_onnx_path", default_value=config.ONNX_MODEL_PATH,
                               width=340, hint="Path to .onnx model file")
            dpg.add_button(label="Browse", callback=_on_onnx_browse)
        dpg.add_spacer(height=4)
        dpg.add_text("Execution provider:")
        dpg.add_combo(items=["Auto","OpenCL","DirectML","CPU"], tag="set_onnx_provider",
                      default_value=config.ONNX_EXECUTION_PROVIDER.capitalize(), width=140)
        dpg.add_spacer(height=12)
        dpg.add_text("CAPTURE DEVICE", color=(160, 160, 160))
        dpg.add_separator(); dpg.add_spacer(height=4)
        dpg.add_combo(items=_camera_labels, tag="set_camera_combo",
                      default_value=_camera_labels[0] if _camera_labels else "No cameras found",
                      width=300)
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_text("W:")
            dpg.add_input_int(tag="set_cap_width",  default_value=config.CAPTURE_WIDTH,
                              width=76, min_value=320, max_value=3840)
            dpg.add_spacer(width=6)
            dpg.add_text("H:")
            dpg.add_input_int(tag="set_cap_height", default_value=config.CAPTURE_HEIGHT,
                              width=76, min_value=240, max_value=2160)
            dpg.add_spacer(width=6)
            dpg.add_text("FPS:")
            dpg.add_input_int(tag="set_cap_fps",    default_value=config.CAPTURE_FPS,
                              width=60, min_value=1, max_value=120)
        dpg.add_spacer(height=12)
        dpg.add_text("APPEARANCE", color=(160, 160, 160))
        dpg.add_separator(); dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_text("Theme:")
            dpg.add_combo(items=["Dark","Light","Auto"], tag="set_theme",
                          default_value=getattr(config,"THEME","dark").capitalize(), width=120)
        dpg.add_spacer(height=18)
        dpg.add_separator(); dpg.add_spacer(height=8)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Save",   width=110, height=30, callback=_on_settings_save)
            dpg.add_spacer(width=8)
            dpg.add_button(label="Cancel", width=110, height=30, callback=_on_settings_cancel)


def _build_help_popup() -> None:
    with dpg.window(label="Help — OpenMFC", tag="help_popup",
                    modal=True, show=False, width=440, height=420,
                    no_resize=True, pos=(180, 80)):
        dpg.add_text("OpenMFC — Open Multi-Frame Compounding", color=(200, 200, 200))
        dpg.add_separator(); dpg.add_spacer(height=6)
        dpg.add_text("LIVE CAPTURE", color=(160, 160, 160))
        dpg.add_text("1. Press 'Connect Camera' (OPTIONS) to open the live feed.")
        dpg.add_text("2. Live Capture → Start unlocks once feed is active.")
        dpg.add_text("3. Press Start to capture a burst and run the pipeline.")
        dpg.add_spacer(height=6)
        dpg.add_text("MANUAL INPUT", color=(160, 160, 160))
        dpg.add_text("1. Click Browse Files (left panel) to load burst JPEGs.")
        dpg.add_text("2. Press MFNR (with Input) → Start to process.")
        dpg.add_spacer(height=6)
        dpg.add_text("VIEWFINDER", color=(160, 160, 160))
        dpg.add_text("Shows: live camera feed, pipeline output, or any loaded image.")
        dpg.add_text("Click any thumbnail in Project Folder to preview it.")
        dpg.add_text("Use 'Open Output File' (OPTIONS) to browse and preview any output.")
        dpg.add_spacer(height=6)
        dpg.add_text("HISTOGRAM", color=(160, 160, 160))
        dpg.add_text("RGB overlay — updates live from camera, pipeline result, or file load.")
        dpg.add_spacer(height=6)
        dpg.add_text("OPTIONS", color=(160, 160, 160))
        dpg.add_text("ECC Threshold — frame rejection sensitivity.")
        dpg.add_text("HDR Auto — pipeline decides based on scene brightness.")
        dpg.add_text("Enhance — requires ONNX model set in Settings.")
        dpg.add_spacer(height=10)
        dpg.add_separator(); dpg.add_spacer(height=6)
        dpg.add_button(label="Close", width=100, height=28,
                       callback=lambda s, a: dpg.configure_item("help_popup", show=False))


# ─────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────

def build_ui(onnx_session=None, camera_labels: list[str] = None) -> None:
    global _onnx_session, _camera_labels
    _onnx_session  = onnx_session
    _camera_labels = camera_labels or []

    dpg.create_context()
    _init_viewfinder_texture()
    _apply_theme(getattr(config, "THEME", "dark"))

    VP_W, VP_H = 1040, 620
    dpg.create_viewport(
        title="OpenMFC - Multi-Frame Compounding",
        width=VP_W, height=VP_H,
        min_width=780, min_height=560,
    )

    _build_settings_popup()
    _build_help_popup()

    g = _geometry(VP_W, VP_H)

    with dpg.window(tag="main_window", no_title_bar=True, no_move=True,
                    no_scrollbar=True, no_scroll_with_mouse=True):

        # ── Top bar ──────────────────────────────────────────────────
        TITLE   = "OpenMFC  (Open Multi-Frame Compounding)"
        BTN_END = 184
        TITLE_W = len(TITLE) * 7
        _ts = max(4, LEFT_W + g["center_w"] // 2 - BTN_END - TITLE_W // 2)

        with dpg.group(horizontal=True):
            dpg.add_button(label="HOME",     width=60,  height=24, callback=lambda s,a: None)
            dpg.add_button(label="SETTINGS", width=72,  height=24, callback=_on_settings_open)
            dpg.add_button(label="HELP",     width=50,  height=24, callback=_on_help_open)
            dpg.add_spacer(width=_ts, tag="title_spacer")
            dpg.add_text(TITLE, tag="title_text")

        dpg.add_spacer(height=4)

        # ── Body ─────────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ── LEFT COLUMN ──────────────────────────────────────────
            with dpg.group(horizontal=False):

                with dpg.child_window(tag="left_top", width=LEFT_W,
                                      height=g["left_top"], border=True,
                                      no_scrollbar=True):
                    dpg.add_text("Project Folder", color=(200, 200, 200))
                    dpg.add_separator(); dpg.add_spacer(height=3)
                    dpg.add_text(config.OUTPUT_PATH, tag="project_path_label",
                                 color=(110, 110, 110), wrap=LEFT_W - 22)
                    dpg.add_spacer(height=4)
                    with dpg.child_window(tag="project_gallery",
                                          width=LEFT_W - 20,
                                          height=max(40, g["left_top"] - 94),
                                          border=False):
                        dpg.add_text("No outputs yet.", color=(90, 90, 90))
                    dpg.add_spacer(height=4)
                    dpg.add_button(label="Refresh Gallery", width=-1, height=24,
                                   callback=lambda s, a: _refresh_project_folder())

                dpg.add_spacer(height=GAP)

                with dpg.child_window(tag="left_bot", width=LEFT_W,
                                      height=g["left_bot"], border=True,
                                      no_scrollbar=True):
                    dpg.add_text("Input", color=(200, 200, 200))
                    dpg.add_separator(); dpg.add_spacer(height=5)
                    with dpg.child_window(tag="input_file_box",
                                          width=LEFT_W - 20,
                                          height=max(40, g["left_bot"] - 74),
                                          border=True):
                        dpg.add_text("No files loaded.", color=(90, 90, 90))
                        dpg.add_spacer(height=4)
                        dpg.add_text("(click Browse to load)", tag="input_label",
                                     color=(90, 90, 90), wrap=LEFT_W - 32)
                    dpg.add_spacer(height=5)
                    dpg.add_button(label="Browse Files", width=-1, height=24,
                                   callback=_on_input_browse)

            dpg.add_spacer(width=2)

            # ── CENTER COLUMN ────────────────────────────────────────
            with dpg.child_window(tag="center_col", width=g["center_w"],
                                  height=g["body_h"], border=False,
                                  no_scrollbar=True, no_scroll_with_mouse=True):

                # Histogram strip
                with dpg.child_window(tag="hist_panel", width=g["center_w"] - 2,
                                      height=HIST_H, border=True, no_scrollbar=True):
                    dpg.add_text("Histogram", color=(100, 100, 100))
                    dpg.add_drawlist(tag="hist_draw",
                                     width=g["center_w"] - 20,
                                     height=HIST_H - 26)

                dpg.add_spacer(height=GAP)

                # Viewfinder
                with dpg.child_window(tag="vf_panel", width=g["center_w"] - 2,
                                      height=g["vf_h"], border=True, no_scrollbar=True):
                    dpg.add_image("vf_texture", tag="vf_image",
                                  width=g["center_w"] - 18,
                                  height=g["vf_h"] - 10)

                dpg.add_spacer(height=GAP)

                # Bottom row: Log + Action buttons — height is fixed LOG_H
                with dpg.group(horizontal=True):

                    with dpg.child_window(tag="log_panel",
                                          width=g["center_w"] - 152,
                                          height=LOG_H, border=True,
                                          no_scrollbar=True):
                        dpg.add_text("Log:", color=(120, 120, 120))
                        dpg.add_input_text(tag="log_box", multiline=True,
                                           readonly=True, width=-1,
                                           height=LOG_H - 36, default_value="")

                    dpg.add_spacer(width=2)

                    with dpg.child_window(tag="btn_panel", width=146, height=LOG_H,
                                          border=True, no_scrollbar=True):
                        dpg.add_text("Live Capture:", color=(170, 170, 170))
                        dpg.add_button(label="Start", tag="live_capture_btn",
                                       width=-1, height=22,
                                       callback=_on_live_capture_start,
                                       enabled=False)
                        dpg.add_spacer(height=2)
                        dpg.add_separator()
                        dpg.add_spacer(height=2)
                        dpg.add_text("MFNR (with Input):", color=(170, 170, 170))
                        dpg.add_button(label="Start", tag="mfnr_input_btn",
                                       width=-1, height=22,
                                       callback=_on_mfnr_input_start)

            dpg.add_spacer(width=2)

            # ── RIGHT COLUMN (OPTIONS) ───────────────────────────────
            # Cards are collapsible headers — zero fixed heights, no clipping ever.
            # Panel scrolls if window is too short.
            with dpg.child_window(tag="right_panel", width=RIGHT_W - 2,
                                  height=g["body_h"], border=True,
                                  no_scrollbar=False):
                dpg.add_text("OPTIONS", color=(200, 200, 200))
                dpg.add_separator(); dpg.add_spacer(height=2)

                # MFNR
                with dpg.collapsing_header(label="MFNR", default_open=True):
                    dpg.add_spacer(height=2)
                    dpg.add_text("ECC Threshold:", color=(140, 140, 140))
                    dpg.add_input_float(tag="ecc_input",
                                        default_value=config.ECC_THRESHOLD,
                                        width=-1, min_value=0.0, max_value=1.0,
                                        step=0.0001, format="%.4f",
                                        callback=_on_ecc_changed)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # HDR
                with dpg.collapsing_header(label="HDR", default_open=True):
                    dpg.add_spacer(height=2)
                    dpg.add_radio_button(items=["Auto", "Enable", "Disable"],
                                         tag="hdr_radio",
                                         default_value=config.HDR_MODE.capitalize(),
                                         callback=_on_hdr_radio)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # Enhance
                with dpg.collapsing_header(label="Enhance", default_open=True):
                    dpg.add_spacer(height=2)
                    dpg.add_radio_button(items=["Enable", "Disable"],
                                         tag="enhance_radio",
                                         default_value="Enable" if config.ENABLE_ONNX else "Disable",
                                         callback=_on_enhance_radio)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # Burst Count
                with dpg.collapsing_header(label="Burst Count", default_open=True):
                    dpg.add_spacer(height=2)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Burst:")
                        dpg.add_input_int(tag="burst_input",
                                          default_value=config.BURST_COUNT,
                                          width=-1, min_value=1, max_value=100,
                                          callback=_on_burst_changed)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # Camera Preview
                with dpg.collapsing_header(label="Camera Preview", default_open=True):
                    dpg.add_spacer(height=2)
                    dpg.add_button(label="Connect Camera", tag="connect_camera_btn",
                                   width=-1, height=24,
                                   callback=_on_connect_camera)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # Viewfinder Ratio
                with dpg.collapsing_header(label="Viewfinder Ratio", default_open=True):
                    dpg.add_spacer(height=2)
                    with dpg.group(horizontal=True):
                        dpg.add_radio_button(items=["Free", "4:3"],
                                             tag="ar_radio",
                                             default_value="Free",
                                             callback=_on_aspect_radio)
                        dpg.add_spacer(width=6)
                        dpg.add_radio_button(items=["16:9", "1:1"],
                                             tag="ar_radio_b",
                                             default_value="",
                                             callback=_on_aspect_radio_b)
                    dpg.add_spacer(height=4)

                dpg.add_spacer(height=2)

                # View Output
                with dpg.collapsing_header(label="View Output", default_open=True):
                    dpg.add_spacer(height=2)
                    dpg.add_button(label="Open Output File", tag="open_output_btn",
                                   width=-1, height=24,
                                   callback=_on_open_output_file)
                    dpg.add_spacer(height=4)

    # ── Finalize ────────────────────────────────────────────────────
    dpg.set_viewport_resize_callback(_on_viewport_resize)
    dpg.set_primary_window("main_window", True)
    dpg.setup_dearpygui()

    import os as _os
    _ico  = _os.path.join("assets", "icon.ico")
    _png  = _os.path.join("assets", "OpenMFC ICON.png")
    _icon = _ico if _os.path.exists(_ico) else (_png if _os.path.exists(_png) else None)
    if _icon:
        try:
            dpg.set_viewport_small_icon(_icon)
            dpg.set_viewport_large_icon(_icon)
            print(f"  Icon: {_icon}")
        except Exception as e:
            print(f"  Icon skipped: {e}")
    else:
        print("  No icon in assets/")

    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        _flush_pending_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()