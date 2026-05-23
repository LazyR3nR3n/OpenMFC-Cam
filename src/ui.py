import os
import sys
import cv2
import threading
import time
import config
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QRadioButton,
    QButtonGroup, QFrame, QSizePolicy, QFileDialog, QScrollArea,
    QTextEdit, QSpinBox, QDialog, QGridLayout, QLineEdit, QProgressBar,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal, QMutex, QMutexLocker, QPoint, QRect
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QColor, QBrush, QIcon

from devices import get_index_from_label
from pipeline import run_pipeline_thread, capture_burst, preview_loop

def _asset_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ─────────────────────────────────────────
#  Stylesheet
# ─────────────────────────────────────────

APP_STYLE = """
QWidget {
    background-color: #1a1a1a;
    color: #e8e8e6;
    font-family: "JetBrains Mono", "Courier New", monospace;
    font-size: 9pt;
}
QMainWindow { background-color: #161616; }

QPushButton {
    background: #303030;
    border: 1px solid #505050;
    color: #c8c8c4;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 1px;
    font-size: 8pt;
    min-height: 22px;
}
QPushButton:hover  { background: #3a3a3a; color: #f0f0ec; border-color: #c8c8c4; }
QPushButton:pressed { background: #282828; }
QPushButton:disabled { color: #444440; border-color: #303030; }

QPushButton#btn_start_live {
    background: #303030; border: 1px solid #505050; color: #f0f0ec;
}
QPushButton#btn_start_mfnr {
    background: #303030; border: 1px solid #585854; color: #c8c8c4;
}
QPushButton#btn_start_live:hover,
QPushButton#btn_start_mfnr:hover {
    background: #c8c8c4; color: #111; border-color: #c8c8c4;
}

QPushButton#titlebar_btn {
    background: none; border: none; color: #666662;
    padding: 2px 10px; font-size: 8pt; letter-spacing: 2px;
    min-height: 24px; border-radius: 0;
}
QPushButton#titlebar_btn:hover { background: #2a2a2a; color: #c8c8c4; }

QPushButton#ev_pill {
    background: #1a1a1a; border: 1px solid #3a3a3a;
    color: #666662; padding: 2px 6px;
    font-size: 8pt; letter-spacing: 0; min-height: 18px;
}
QPushButton#ev_pill:checked {
    background: #323232; border-color: #c8c8c4; color: #f0f0ec;
}

QSlider::groove:horizontal {
    height: 2px; background: #3a3a3a; border-radius: 1px;
}
QSlider::handle:horizontal {
    background: #c8c8c4; width: 10px; height: 10px;
    margin: -4px 0; border-radius: 5px;
}
QSlider::sub-page:horizontal { background: #7a7a76; border-radius: 1px; }

QRadioButton { color: #9a9a96; spacing: 5px; font-size: 8pt; }
QRadioButton::indicator {
    width: 8px; height: 8px; border-radius: 4px;
    border: 1px solid #505050; background: transparent;
}
QRadioButton::indicator:checked { background: #c8c8c4; border-color: #c8c8c4; }

QComboBox {
    background: #1a1a1a; border: 1px solid #3a3a3a;
    color: #9a9a96; padding: 2px 5px; border-radius: 2px; font-size: 8pt;
}
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background: #212121; border: 1px solid #3a3a3a;
    color: #9a9a96; selection-background-color: #2a2a2a;
}

QLineEdit, QTextEdit {
    background: #0e0e0e; border: 1px solid #3a3a3a;
    color: #9a9a96; padding: 3px 5px; border-radius: 2px; font-size: 8pt;
}

QSpinBox {
    background: #1a1a1a; border: 1px solid #3a3a3a;
    color: #c8c8c4; padding: 2px 4px; border-radius: 2px; font-size: 9pt;
}
QSpinBox::up-button, QSpinBox::down-button { background: #2a2a2a; border: none; width: 16px; }

QScrollBar:vertical { background: #1a1a1a; width: 6px; margin: 0; }
QScrollBar::handle:vertical { background: #3a3a3a; border-radius: 3px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #505050; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #3a3a3a; }
QDialog { background: #1e1e1e; }
"""


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────

def _h_rule() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Plain)
    f.setStyleSheet("color: #3a3a3a;")
    return f


def _muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #666662; font-size: 8pt;")
    return lbl


def _opt_header(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "background: #252525; color: #c8c8c4; letter-spacing: 2px; "
        "padding: 4px 10px; border-bottom: 1px solid #3a3a3a; "
        "border-top: 1px solid #3a3a3a; font-size: 8pt; font-weight: bold;"
    )
    return lbl


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(
        "color: #9a9a96; font-size: 8pt; letter-spacing: 2px; "
        "background: #2a2a2a; padding: 4px 8px; border-bottom: 1px solid #3a3a3a;"
    )
    return lbl


# ─────────────────────────────────────────
#  Histogram PiP (draggable overlay)
# ─────────────────────────────────────────

class HistogramPip(QWidget):
    MODES = ["Luma", "RGB", "All"]

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(162, 112)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "background: rgba(14,14,14,220); border: 1px solid #3a3a3a; border-radius: 3px;"
        )
        self._mode = "Luma"
        self._hist_data: dict = {}
        self._drag_pos: QPoint | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar / drag handle
        bar = QWidget()
        bar.setFixedHeight(22)
        bar.setStyleSheet("background: rgba(28,28,28,240); border-bottom: 1px solid #2a2a2a;")
        bar_l = QHBoxLayout(bar)
        bar_l.setContentsMargins(6, 0, 4, 0)
        bar_l.setSpacing(3)
        title = QLabel("Histogram")
        title.setStyleSheet("color: #666662; font-size: 7pt; letter-spacing: 2px; background: transparent; border: none;")
        bar_l.addWidget(title)
        bar_l.addStretch()

        self._mode_btns: list[QPushButton] = []
        for m in self.MODES:
            btn = QPushButton(m)
            btn.setCheckable(True)
            btn.setFixedHeight(16)
            btn.setStyleSheet(
                "QPushButton { background: transparent; border: 1px solid #3a3a3a; "
                "color: #666662; font-size: 7pt; padding: 0 4px; border-radius: 1px; "
                "letter-spacing: 0; min-height: 0; } "
                "QPushButton:checked { border-color: #9a9a96; color: #c8c8c4; } "
                "QPushButton:hover   { color: #9a9a96; }"
            )
            btn.clicked.connect(lambda _, mode=m: self._set_mode(mode))
            bar_l.addWidget(btn)
            self._mode_btns.append(btn)
        self._mode_btns[0].setChecked(True)
        root.addWidget(bar)

        # Canvas
        self._canvas = QWidget()
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet("background: transparent; border: none;")
        self._canvas.paintEvent = self._paint_histogram  # type: ignore
        root.addWidget(self._canvas)

        # Axis
        axis = QWidget()
        axis.setFixedHeight(14)
        axis.setStyleSheet("background: transparent; border: none;")
        ax_l = QHBoxLayout(axis)
        ax_l.setContentsMargins(6, 0, 6, 2)
        ax_l.setSpacing(0)
        for v in ("0", "64", "128", "192", "255"):
            lbl = QLabel(v)
            lbl.setStyleSheet("color: #444440; font-size: 6pt; background: transparent; border: none;")
            ax_l.addWidget(lbl)
            if v != "255":
                ax_l.addStretch()
        root.addWidget(axis)

        # Default position: bottom-right of parent
        if parent:
            self.move(parent.width() - self.width() - 10, parent.height() - self.height() - 10)

    def _set_mode(self, mode: str):
        self._mode = mode
        for btn in self._mode_btns:
            btn.setChecked(btn.text() == mode)
        self._canvas.update()

    def update_frame(self, frame: np.ndarray):
        bins = 64
        self._hist_data = {}
        for ch, key in enumerate(["b", "g", "r"]):
            h = cv2.calcHist([frame], [ch], None, [bins], [0, 256])
            cv2.normalize(h, h, 0, 1.0, cv2.NORM_MINMAX)
            self._hist_data[key] = h.flatten()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        cv2.normalize(h, h, 0, 1.0, cv2.NORM_MINMAX)
        self._hist_data["luma"] = h.flatten()
        self._canvas.update()

    def _paint_histogram(self, event=None):
        canvas = self._canvas
        w, h = canvas.width(), canvas.height()
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, w, h, QColor("#0e0e0e"))

        # Grid
        pen = QPen(QColor(60, 60, 60, 100))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        for frac in (0.25, 0.5, 0.75):
            x = int(w * frac)
            painter.drawLine(x, 0, x, h)

        if not self._hist_data:
            painter.end()
            return

        bins = 64
        step = w / bins

        def draw_filled(data, color: QColor, alpha: int = 180):
            from PyQt6.QtCore import QPointF
            color.setAlpha(alpha)
            fill = QColor(color); fill.setAlpha(55)
            pts = [QPointF(0, h)] + \
                  [QPointF(i * step, h - data[i] * (h - 2)) for i in range(bins)] + \
                  [QPointF(w, h)]
            painter.setBrush(QBrush(fill))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(pts)
            lpen = QPen(color); lpen.setWidthF(1.0)
            painter.setPen(lpen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(bins - 1):
                painter.drawLine(
                    int(i * step),     int(h - data[i] * (h - 2)),
                    int((i+1)*step),   int(h - data[i+1] * (h-2)),
                )

        mode = self._mode
        if mode == "Luma" and "luma" in self._hist_data:
            draw_filled(self._hist_data["luma"], QColor(200, 200, 196))
        elif mode == "RGB":
            if "b" in self._hist_data: draw_filled(self._hist_data["b"], QColor(80, 140, 240))
            if "g" in self._hist_data: draw_filled(self._hist_data["g"], QColor(80, 200, 100))
            if "r" in self._hist_data: draw_filled(self._hist_data["r"], QColor(220, 80, 80))
        elif mode == "All":
            if "b" in self._hist_data: draw_filled(self._hist_data["b"], QColor(80, 140, 240), 110)
            if "g" in self._hist_data: draw_filled(self._hist_data["g"], QColor(80, 200, 100), 110)
            if "r" in self._hist_data: draw_filled(self._hist_data["r"], QColor(220, 80, 80), 110)
            if "luma" in self._hist_data:
                d = self._hist_data["luma"]
                lpen = QPen(QColor(200, 200, 196, 210)); lpen.setWidthF(1.2)
                painter.setPen(lpen); painter.setBrush(Qt.BrushStyle.NoBrush)
                for i in range(bins - 1):
                    painter.drawLine(
                        int(i*step), int(h - d[i]*(h-2)),
                        int((i+1)*step), int(h - d[i+1]*(h-2)),
                    )
        painter.end()

    # Drag
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and self.parent():
            delta = e.pos() - self._drag_pos
            new_pos = self.pos() + delta
            p = self.parent()
            new_pos.setX(max(0, min(new_pos.x(), p.width()  - self.width())))
            new_pos.setY(max(0, min(new_pos.y(), p.height() - self.height())))
            self.move(new_pos)

    def mouseReleaseEvent(self, e):
        self._drag_pos = None


# ─────────────────────────────────────────
#  Viewfinder
# ─────────────────────────────────────────

_ASPECT_RATIOS: dict[str, tuple[int, int] | None] = {
    "Free": None,
    "16:9": (16, 9),
    "4:3":  (4,  3),
    "1:1":  (1,  1),
    "3:2":  (3,  2),
}


class ViewfinderWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background: #0e0e0e;")
        self._pixmap: QPixmap | None = None
        self._aspect: tuple[int, int] | None = None   # None = Free

        self.hist_pip = HistogramPip(self)
        self.hist_pip.raise_()

    def set_aspect(self, ratio_str: str):
        if ratio_str.startswith("custom:"):
            try:
                _, w, h = ratio_str.split(":")
                self._aspect = (int(w), int(h))
            except (ValueError, TypeError):
                self._aspect = None
        else:
            self._aspect = _ASPECT_RATIOS.get(ratio_str, None)
        self.update()

    def _constrained_rect(self) -> QRect:
        """Return the display rect constrained to the current aspect ratio."""
        if self._aspect is None:
            return self.rect()
        aw, ah = self._aspect
        w, h = self.width(), self.height()
        # Fit ratio inside widget
        if w / h > aw / ah:
            # Widget is wider than ratio — constrain by height
            nw = int(h * aw / ah)
            return QRect((w - nw) // 2, 0, nw, h)
        else:
            nh = int(w * ah / aw)
            return QRect(0, (h - nh) // 2, w, nh)


    def resizeEvent(self, e):
        super().resizeEvent(e)
        pip = self.hist_pip
        x = min(pip.x(), self.width()  - pip.width())
        y = min(pip.y(), self.height() - pip.height())
        pip.move(max(0, x), max(0, y))

    def set_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self.hist_pip.update_frame(frame)
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e0e0e"))

        # Grid
        pen = QPen(QColor(255, 255, 255, 8))
        pen.setWidthF(0.5)
        painter.setPen(pen)
        for x in range(0, self.width(), 30):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 30):
            painter.drawLine(0, y, self.width(), y)

        if self._pixmap:
            target = self._constrained_rect()
            scaled = self._pixmap.scaled(
                target.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = target.x() + (target.width()  - scaled.width())  // 2
            y = target.y() + (target.height() - scaled.height()) // 2
            # Darken letterbox bars if aspect is constrained
            if self._aspect is not None:
                painter.fillRect(self.rect(), QColor("#0e0e0e"))
            painter.drawPixmap(x, y, scaled)
        else:
            # Corner brackets
            pen = QPen(QColor("#505050")); pen.setWidthF(1.0)
            painter.setPen(pen)
            L = 14
            for cx, cy, left, top in [
                (8, 8, True, True),
                (self.width()-8, 8, False, True),
                (8, self.height()-8, True, False),
                (self.width()-8, self.height()-8, False, False),
            ]:
                dx = L if left else -L
                dy = L if top else -L
                painter.drawLine(cx, cy, cx+dx, cy)
                painter.drawLine(cx, cy, cx, cy+dy)

            # Crosshair
            cx, cy = self.width()//2, self.height()//2
            painter.drawLine(cx, cy-20, cx, cy+20)
            painter.drawLine(cx-20, cy, cx+20, cy)

            # Label
            painter.setPen(QColor("#444440"))
            f = QFont("JetBrains Mono", 8)
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
            painter.setFont(f)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                "  AWAITING CAMERA OR INPUT",
            )
        painter.end()


# ─────────────────────────────────────────
#  Frame signaler (thread → main)
# ─────────────────────────────────────────

class FrameSignaler(QObject):
    frame_ready      = pyqtSignal(object)
    clear_view       = pyqtSignal()
    cameras_ready    = pyqtSignal(list)   # emitted from bg thread when enumeration finishes
    progress_update  = pyqtSignal(int)    # 0-100, emitted by pipeline thread


# ─────────────────────────────────────────
#  Left panel
# ─────────────────────────────────────────

class LeftPanel(QWidget):
    image_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(160)
        self.setMaximumWidth(185)
        self.setStyleSheet("background: #1e1e1e; border-right: 1px solid #3a3a3a;")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Project Folder
        root.addWidget(_section_label("📁  Project Folder"))
        fb = QWidget()
        fl = QVBoxLayout(fb)
        fl.setContentsMargins(8, 8, 8, 8)
        fl.setSpacing(6)
        self.lbl_output_path = QLabel(config.OUTPUT_PATH)
        self.lbl_output_path.setWordWrap(True)
        self.lbl_output_path.setStyleSheet(
            "background: #0e0e0e; border: 1px solid #3a3a3a; color: #666662; "
            "padding: 5px; font-size: 8pt; border-radius: 2px; min-height: 44px;"
        )
        fl.addWidget(self.lbl_output_path)
        self.btn_change_output = QPushButton("Change Folder")
        fl.addWidget(self.btn_change_output)
        root.addWidget(fb)
        root.addWidget(_h_rule())

        # Gallery
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.gallery_scroll.setStyleSheet("border: none; background: #1a1a1a;")
        self.gallery_inner = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_inner)
        self.gallery_layout.setContentsMargins(6, 6, 6, 6)
        self.gallery_layout.setSpacing(4)
        self.gallery_layout.addWidget(_muted("No outputs yet."))
        self.gallery_layout.addStretch()
        self.gallery_scroll.setWidget(self.gallery_inner)
        root.addWidget(self.gallery_scroll, 1)

        btn_refresh = QPushButton("Refresh Gallery")
        btn_refresh.setStyleSheet("border-radius: 0; border-top: 1px solid #3a3a3a;")
        btn_refresh.clicked.connect(self.refresh_gallery)
        root.addWidget(btn_refresh)
        root.addWidget(_h_rule())

        # Input
        root.addWidget(_section_label("📷  Input"))
        ib = QWidget()
        il = QVBoxLayout(ib)
        il.setContentsMargins(8, 8, 8, 8)
        il.setSpacing(6)
        self.lbl_drop = QLabel("Drop frames here\nor open folder")
        self.lbl_drop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_drop.setStyleSheet(
            "background: #0e0e0e; border: 1px dashed #3a3a3a; color: #666662; "
            "padding: 10px; font-size: 8pt; border-radius: 2px; min-height: 52px;"
        )
        il.addWidget(self.lbl_drop)
        self.btn_open_folder = QPushButton("Open Folder")
        il.addWidget(self.btn_open_folder)
        root.addWidget(ib)

    def refresh_gallery(self):
        # Clear
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        folder = Path(config.OUTPUT_PATH)
        if not folder.exists():
            self.gallery_layout.addWidget(_muted("Folder not found."))
            self.gallery_layout.addStretch()
            return

        exts = {".jpg", ".jpeg", ".png"}
        images = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in exts],
            key=lambda f: f.stat().st_mtime, reverse=True,
        )[:20]

        if not images:
            self.gallery_layout.addWidget(_muted("No outputs yet."))
            self.gallery_layout.addStretch()
            return

        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            thumb = cv2.resize(img, (148, 84))
            rgb   = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            qi    = QImage(rgb.data, 148, 84, rgb.strides[0], QImage.Format.Format_RGB888)
            pix   = QPixmap.fromImage(qi)

            btn = QPushButton()
            btn.setIcon(QIcon(pix))
            btn.setIconSize(QSize(148, 84))
            btn.setFixedSize(154, 90)
            btn.setStyleSheet(
                "border: 1px solid #3a3a3a; background: #0e0e0e; "
                "border-radius: 2px; padding: 2px;"
            )
            path_str = str(p)
            btn.clicked.connect(lambda _, ps=path_str: self.image_selected.emit(ps))
            self.gallery_layout.addWidget(btn)
            self.gallery_layout.addWidget(_muted(p.name[:18]))

        self.gallery_layout.addStretch()


# ─────────────────────────────────────────
#  Options panel (right rail)
# ─────────────────────────────────────────

class OptionsPanel(QWidget):
    ecc_changed        = pyqtSignal(float)
    hdr_mode_changed   = pyqtSignal(str)
    ev_bracket_changed = pyqtSignal(str)
    fusion_changed     = pyqtSignal(float)
    enhance_changed    = pyqtSignal(str)
    burst_changed      = pyqtSignal(int)
    aspect_changed     = pyqtSignal(str)
    camera_changed     = pyqtSignal(str)
    connect_camera     = pyqtSignal()
    open_output        = pyqtSignal()

    def __init__(self, camera_labels: list[str], parent=None):
        super().__init__(parent)
        self.setMinimumWidth(190)
        self.setMaximumWidth(220)
        self.setStyleSheet("background: #1e1e1e; border-left: 1px solid #3a3a3a;")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("border: none; background: transparent;")

        inner = QWidget()
        inner.setStyleSheet("background: #1e1e1e;")
        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        h = QLabel("OPTIONS")
        h.setStyleSheet(
            "background: #161616; color: #666662; letter-spacing: 4px; "
            "padding: 5px 10px; border-bottom: 1px solid #3a3a3a; font-size: 8pt;"
        )
        root.addWidget(h)

        # ── MFNR ──────────────────────────────
        root.addWidget(_opt_header("MFNR"))
        mb = QWidget(); ml = QVBoxLayout(mb)
        ml.setContentsMargins(10, 6, 10, 8); ml.setSpacing(4)
        ecc_row = QHBoxLayout()
        ecc_row.addWidget(_muted("ECC Threshold"))
        self.lbl_ecc = QLabel(f"{config.ECC_THRESHOLD:.1f}")
        self.lbl_ecc.setStyleSheet("color: #9a9a96; font-size: 8pt;")
        ecc_row.addStretch(); ecc_row.addWidget(self.lbl_ecc)
        ml.addLayout(ecc_row)
        self.sld_ecc = QSlider(Qt.Orientation.Horizontal)
        self.sld_ecc.setRange(0, 10)
        self.sld_ecc.setValue(int(config.ECC_THRESHOLD * 10))
        self.sld_ecc.valueChanged.connect(self._on_ecc)
        ml.addWidget(self.sld_ecc)
        root.addWidget(mb); root.addWidget(_h_rule())

        # ── HDR ───────────────────────────────
        root.addWidget(_opt_header("HDR"))
        hb = QWidget(); hl = QVBoxLayout(hb)
        hl.setContentsMargins(10, 6, 10, 8); hl.setSpacing(5)

        self._hdr_grp = QButtonGroup(self)
        for mode in ("Auto", "Enable", "Disable"):
            rb = QRadioButton(mode)
            if mode.lower() == getattr(config, "HDR_MODE", "auto"):
                rb.setChecked(True)
            self._hdr_grp.addButton(rb)
            hl.addWidget(rb)
        self._hdr_grp.buttonClicked.connect(
            lambda btn: self.hdr_mode_changed.emit(btn.text())
        )

        hl.addSpacing(4)
        hl.addWidget(_muted("EV Bracket"))
        ev_row = QHBoxLayout(); ev_row.setSpacing(3)
        self._ev_grp = QButtonGroup(self)
        self._ev_grp.setExclusive(True)
        for label in ("±1", "±2", "±3", "Custom"):
            btn = QPushButton(label)
            btn.setObjectName("ev_pill")
            btn.setCheckable(True)
            if label == "±1":
                btn.setChecked(True)
            self._ev_grp.addButton(btn)
            ev_row.addWidget(btn)
        self._ev_grp.buttonClicked.connect(
            lambda btn: self.ev_bracket_changed.emit(btn.text())
        )
        hl.addLayout(ev_row)

        hl.addSpacing(3)
        fw_row = QHBoxLayout()
        fw_row.addWidget(_muted("Fusion Weight"))
        self.lbl_fw = QLabel("0.5")
        self.lbl_fw.setStyleSheet("color: #9a9a96; font-size: 8pt;")
        fw_row.addStretch(); fw_row.addWidget(self.lbl_fw)
        hl.addLayout(fw_row)
        self.sld_fusion = QSlider(Qt.Orientation.Horizontal)
        self.sld_fusion.setRange(0, 10)
        self.sld_fusion.setValue(5)
        self.sld_fusion.valueChanged.connect(self._on_fusion)
        hl.addWidget(self.sld_fusion)
        axis_row = QHBoxLayout()
        axis_row.addWidget(_muted("Shadows"))
        axis_row.addStretch()
        axis_row.addWidget(_muted("Highlights"))
        hl.addLayout(axis_row)

        root.addWidget(hb); root.addWidget(_h_rule())

        # ── Enhance ───────────────────────────
        root.addWidget(_opt_header("Enhance"))
        eb = QWidget(); el = QVBoxLayout(eb)
        el.setContentsMargins(10, 6, 10, 8); el.setSpacing(3)
        self._enh_grp = QButtonGroup(self)
        for opt in ("Enable", "Disable"):
            rb = QRadioButton(opt)
            enabled = getattr(config, "ENABLE_ONNX", False)
            if (opt == "Enable" and enabled) or (opt == "Disable" and not enabled):
                rb.setChecked(True)
            self._enh_grp.addButton(rb)
            el.addWidget(rb)
        self._enh_grp.buttonClicked.connect(
            lambda btn: self.enhance_changed.emit(btn.text())
        )
        root.addWidget(eb); root.addWidget(_h_rule())

        # ── Burst Count ───────────────────────
        root.addWidget(_opt_header("Burst Count"))
        bb = QWidget(); bl = QHBoxLayout(bb)
        bl.setContentsMargins(10, 6, 10, 8); bl.setSpacing(8)
        self.spin_burst = QSpinBox()
        self.spin_burst.setRange(1, 64)
        self.spin_burst.setValue(config.BURST_COUNT)
        self.spin_burst.setFixedWidth(64)
        self.spin_burst.valueChanged.connect(self.burst_changed.emit)
        bl.addWidget(self.spin_burst)
        bl.addWidget(_muted("frames"))
        bl.addStretch()
        root.addWidget(bb); root.addWidget(_h_rule())

        # ── Camera ────────────────────────────
        root.addWidget(_opt_header("Camera"))
        cb = QWidget(); cl = QVBoxLayout(cb)
        cl.setContentsMargins(10, 6, 10, 8); cl.setSpacing(6)
        self.cmb_camera = QComboBox()
        self.cmb_camera.addItems(camera_labels if camera_labels else ["No device found"])
        self.cmb_camera.currentTextChanged.connect(self.camera_changed.emit)
        cl.addWidget(self.cmb_camera)
        self.btn_connect = QPushButton("Connect Camera")
        self.btn_connect.clicked.connect(self.connect_camera.emit)
        cl.addWidget(self.btn_connect)
        root.addWidget(cb); root.addWidget(_h_rule())

        # ── Aspect Ratio ──────────────────────
        root.addWidget(_opt_header("Aspect Ratio"))
        ab = QWidget(); al = QVBoxLayout(ab)
        al.setContentsMargins(10, 6, 10, 8); al.setSpacing(3)
        self._ar_grp = QButtonGroup(self)
        for ratio in ("Free", "16:9", "4:3", "1:1", "3:2", "Custom"):
            rb = QRadioButton(ratio)
            if ratio == "Free":
                rb.setChecked(True)
            self._ar_grp.addButton(rb)
            al.addWidget(rb)
        self._ar_grp.buttonClicked.connect(self._on_aspect_btn)

        # Custom W:H input — hidden until "Custom" is selected
        self._custom_ar_row = QWidget()
        self._custom_ar_row.setVisible(False)
        cr = QHBoxLayout(self._custom_ar_row)
        cr.setContentsMargins(0, 2, 0, 0); cr.setSpacing(4)
        self._sp_ar_w = QSpinBox(); self._sp_ar_w.setRange(1, 9999); self._sp_ar_w.setValue(16)
        self._sp_ar_w.setFixedWidth(52)
        cr.addWidget(self._sp_ar_w)
        cr.addWidget(_muted(":"))
        self._sp_ar_h = QSpinBox(); self._sp_ar_h.setRange(1, 9999); self._sp_ar_h.setValue(9)
        self._sp_ar_h.setFixedWidth(52)
        cr.addWidget(self._sp_ar_h)
        btn_apply_ar = QPushButton("Apply")
        btn_apply_ar.setFixedHeight(20)
        btn_apply_ar.setStyleSheet("font-size: 7pt; padding: 0 4px; min-height: 0;")
        btn_apply_ar.clicked.connect(self._on_custom_ar_apply)
        cr.addWidget(btn_apply_ar)
        al.addWidget(self._custom_ar_row)

        root.addWidget(ab); root.addWidget(_h_rule())

        # ── Open Output ───────────────────────
        root.addWidget(_opt_header("View Output"))
        vb = QWidget(); vl = QVBoxLayout(vb)
        vl.setContentsMargins(10, 6, 10, 8)
        btn_open = QPushButton("Open Output File")
        btn_open.clicked.connect(self.open_output.emit)
        vl.addWidget(btn_open)
        root.addWidget(vb)
        root.addStretch()

        scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def _on_ecc(self, val: int):
        v = round(val / 10, 1)
        self.lbl_ecc.setText(f"{v:.1f}")
        self.ecc_changed.emit(v)

    def _on_fusion(self, val: int):
        v = round(val / 10, 1)
        self.lbl_fw.setText(f"{v:.1f}")
        self.fusion_changed.emit(v)

    def _on_aspect_btn(self, btn):
        label = btn.text()
        is_custom = label == "Custom"
        self._custom_ar_row.setVisible(is_custom)
        if not is_custom:
            self.aspect_changed.emit(label)

    def _on_custom_ar_apply(self):
        w = self._sp_ar_w.value()
        h = self._sp_ar_h.value()
        self.aspect_changed.emit(f"custom:{w}:{h}")


# ─────────────────────────────────────────
#  Settings dialog
# ─────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, camera_labels: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings — OpenMFC")
        self.setFixedSize(500, 520)
        self.setStyleSheet(APP_STYLE)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        def section(txt):
            lbl = QLabel(txt.upper())
            lbl.setStyleSheet("color: #9a9a96; font-size: 8pt; letter-spacing: 2px;")
            root.addWidget(lbl)
            root.addWidget(_h_rule())

        section("Output")
        og = QGridLayout(); og.setSpacing(6)
        og.addWidget(_muted("Folder:"), 0, 0)
        self.ed_output = QLineEdit(config.OUTPUT_PATH)
        og.addWidget(self.ed_output, 0, 1)
        b = QPushButton("Browse"); b.setFixedWidth(70)
        b.clicked.connect(lambda: self.ed_output.setText(
            QFileDialog.getExistingDirectory(self, "Output Folder") or self.ed_output.text()
        ))
        og.addWidget(b, 0, 2)
        og.addWidget(_muted("Format:"), 1, 0)
        self.cmb_format = QComboBox(); self.cmb_format.addItems(["JPG", "PNG"])
        self.cmb_format.setCurrentText(config.OUTPUT_FORMAT.upper())
        og.addWidget(self.cmb_format, 1, 1)
        og.addWidget(_muted("Naming:"), 2, 0)
        self.cmb_naming = QComboBox(); self.cmb_naming.addItems(["Timestamp", "Sequential"])
        self.cmb_naming.setCurrentText(config.OUTPUT_NAMING.capitalize())
        og.addWidget(self.cmb_naming, 2, 1)
        root.addLayout(og); root.addSpacing(6)

        section("ONNX Model")
        xg = QGridLayout(); xg.setSpacing(6)
        xg.addWidget(_muted("Model:"), 0, 0)
        self.ed_onnx = QLineEdit(config.ONNX_MODEL_PATH)
        xg.addWidget(self.ed_onnx, 0, 1)
        bx = QPushButton("Browse"); bx.setFixedWidth(70)
        bx.clicked.connect(lambda: self._browse_onnx())
        xg.addWidget(bx, 0, 2)
        xg.addWidget(_muted("Provider:"), 1, 0)
        self.cmb_provider = QComboBox()
        self.cmb_provider.addItems(["Auto", "OpenCL", "DirectML", "CPU"])
        self.cmb_provider.setCurrentText(
            getattr(config, "ONNX_EXECUTION_PROVIDER", "auto").capitalize()
        )
        xg.addWidget(self.cmb_provider, 1, 1)
        xg.addWidget(_muted("Resolution:"), 2, 0)
        self.sp_enhance_max_height = QSpinBox()
        self.sp_enhance_max_height.setRange(480, 2160)
        self.sp_enhance_max_height.setValue(config.ENHANCE_MAX_HEIGHT)
        self.sp_enhance_max_height.setSuffix(" px")
        xg.addWidget(self.sp_enhance_max_height, 2, 1)
        root.addLayout(xg); root.addSpacing(6)

        section("Capture Device")
        cg = QGridLayout(); cg.setSpacing(6)
        cg.addWidget(_muted("Device:"), 0, 0)
        self.cmb_cam = QComboBox()
        self.cmb_cam.addItems(camera_labels if camera_labels else ["No cameras found"])
        cg.addWidget(self.cmb_cam, 0, 1, 1, 2)
        for row, (lbl, attr, lo, hi) in enumerate([
            ("Width:", "CAPTURE_WIDTH", 320, 3840),
            ("Height:", "CAPTURE_HEIGHT", 240, 2160),
            ("FPS:", "CAPTURE_FPS", 1, 120),
        ], start=1):
            cg.addWidget(_muted(lbl), row, 0)
            sp = QSpinBox(); sp.setRange(lo, hi); sp.setValue(getattr(config, attr))
            setattr(self, f"sp_{attr.lower()}", sp)
            cg.addWidget(sp, row, 1)
        root.addLayout(cg)

        root.addStretch()
        root.addWidget(_h_rule())
        root.addSpacing(6)
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save"); btn_save.setFixedHeight(30)
        btn_save.clicked.connect(self._save)
        btn_cancel = QPushButton("Cancel"); btn_cancel.setFixedHeight(30)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_save); btn_row.addSpacing(8)
        btn_row.addWidget(btn_cancel); btn_row.addStretch()
        root.addLayout(btn_row)

    def _browse_onnx(self):
            f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.onnx *.pth);;All (*.*)")
            if f:
                if f.endswith(".pth"):
                    f = self._convert_pth_to_onnx(f)
                if f:
                    self.ed_onnx.setText(f)

    def _convert_pth_to_onnx(self, pth_path: str) -> str:
        import subprocess, sys, os
        onnx_path = os.path.splitext(pth_path)[0] + ".onnx"
        script = os.path.join("assets", "pytorch2onnx.py")

        msg = QMessageBox(self)
        msg.setWindowTitle("Converting Model")
        msg.setText("Converting .pth to .onnx, please wait...")
        msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
        msg.show()
        QApplication.processEvents()

        try:
            subprocess.run(
                [sys.executable, script, "--input", pth_path, "--output", onnx_path],
                check=True
            )
            msg.close()
            return onnx_path
        except subprocess.CalledProcessError:
            msg.close()
            QMessageBox.critical(self, "Conversion Failed", "Could not convert .pth to .onnx. Check that PyTorch is installed.")
            return ""

    def _save(self):
        config.OUTPUT_PATH             = self.ed_output.text()
        config.OUTPUT_FORMAT           = self.cmb_format.currentText().lower()
        config.OUTPUT_NAMING           = self.cmb_naming.currentText().lower()
        config.ONNX_MODEL_PATH         = self.ed_onnx.text()
        config.ONNX_EXECUTION_PROVIDER = self.cmb_provider.currentText().lower()
        config.CAPTURE_DEVICE          = get_index_from_label(self.cmb_cam.currentText())
        config.CAPTURE_WIDTH           = self.sp_capture_width.value()
        config.CAPTURE_HEIGHT          = self.sp_capture_height.value()
        config.CAPTURE_FPS             = self.sp_capture_fps.value()

        import json, os
        os.makedirs(os.path.dirname(config.SETTINGS_PATH), exist_ok=True)
        payload = {
            "OUTPUT_PATH": config.OUTPUT_PATH,
            "OUTPUT_FORMAT": config.OUTPUT_FORMAT,
            "OUTPUT_NAMING": config.OUTPUT_NAMING,
            "ONNX_MODEL_PATH": config.ONNX_MODEL_PATH,
            "ONNX_EXECUTION_PROVIDER": config.ONNX_EXECUTION_PROVIDER,
            "CAPTURE_DEVICE": config.CAPTURE_DEVICE,
            "CAPTURE_WIDTH": config.CAPTURE_WIDTH,
            "CAPTURE_HEIGHT": config.CAPTURE_HEIGHT,
            "CAPTURE_FPS": config.CAPTURE_FPS,
        }
        with open(config.SETTINGS_PATH, "w") as f:
            json.dump(payload, f, indent=2)

        self.accept()

# ─────────────────────────────────────────
#  Help dialog
# ─────────────────────────────────────────

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help — OpenMFC")
        self.setFixedSize(440, 440)
        self.setStyleSheet(APP_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(6)

        title = QLabel("OpenMFC — Open Multi-Frame Compounding")
        title.setStyleSheet("color: #c8c8c4; font-size: 9pt;")
        root.addWidget(title)
        root.addWidget(_h_rule())

        def section(txt):
            lbl = QLabel(txt.upper())
            lbl.setStyleSheet("color: #9a9a96; font-size: 8pt; letter-spacing: 2px; margin-top: 6px;")
            root.addWidget(lbl)

        def item(txt):
            lbl = QLabel(txt)
            lbl.setStyleSheet("color: #666662; font-size: 8pt;")
            lbl.setWordWrap(True)
            root.addWidget(lbl)

        section("Live Capture")
        item("1. Press 'Connect Camera' in Options to open the live feed.")
        item("2. Live Capture → Start unlocks once feed is active.")
        item("3. Press Start to capture a burst and run the pipeline.")

        section("Manual Input")
        item("1. Click Open Folder (left panel) to load burst images.")
        item("2. Press MFNR Start to process the loaded frames.")

        section("Viewfinder")
        item("Shows: live camera feed, pipeline output, or any loaded image.")
        item("Click any thumbnail in Project Folder to preview it.")

        section("Histogram PiP")
        item("Draggable overlay on the viewfinder. Switch: Luma / RGB / All.")
        item("Updates live from camera, pipeline result, or file load.")

        section("Options")
        item("ECC Threshold — frame rejection sensitivity for alignment.")
        item("EV Bracket — exposure stop spread for Mertens HDR fusion.")
        item("Fusion Weight — bias shadows or highlights in the HDR merge.")
        item("Enhance — ONNX model required (set path in Settings).")

        root.addStretch()
        root.addWidget(_h_rule())
        root.addSpacing(6)
        btn = QPushButton("Close"); btn.setFixedWidth(100)
        btn.clicked.connect(self.accept)
        root.addWidget(btn)


# ─────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, onnx_session=None, camera_labels: list[str] = None):
        super().__init__()
        # ... all your existing code ...
        
        # Enumerate cameras in background — avoids blocking the UI on startup
        threading.Thread(target=self._enumerate_cameras_bg, daemon=True).start()

        # Check for updates in background on startup
        from updater import check_for_updates
        check_for_updates(parent=self, silent=True)
        
        self.setWindowTitle("OpenMFC — Multi-Frame Compounding")
        self.setMinimumSize(980, 580)
        self.resize(1200, 700)
        self.setStyleSheet(APP_STYLE)
        self.setWindowIcon(QIcon(_asset_path(os.path.join("assets", "icon.ico"))))

        self._onnx_session    = onnx_session
        self._camera_labels   = camera_labels or []
        self._input_paths:    list[str] = []
        self._preview_running = False
        self._live_active     = False

        self._frame_mutex    = QMutex()
        self._pending_frame: np.ndarray | None = None
        self._frame_dirty    = False

        self._signaler = FrameSignaler()

        self._build_ui()
        self._connect_signals()

        # Wire signaler after viewfinder exists
        self._signaler.frame_ready.connect(self._on_frame_ready)
        self._signaler.clear_view.connect(self.viewfinder.clear)
        self._signaler.cameras_ready.connect(self._on_cameras_ready)
        self._signaler.progress_update.connect(self.progress_bar.setValue)

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(33)   # ~30 fps
        self._flush_timer.timeout.connect(self._flush_pending_frame)
        self._flush_timer.start()

        # Enumerate cameras in background — avoids blocking the UI on startup
        threading.Thread(target=self._enumerate_cameras_bg, daemon=True).start()

    # ─────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Titlebar
        titlebar = QWidget()
        titlebar.setFixedHeight(30)
        titlebar.setStyleSheet("background: #161616; border-bottom: 1px solid #3a3a3a;")
        tb_l = QHBoxLayout(titlebar)
        tb_l.setContentsMargins(0, 0, 0, 0)
        tb_l.setSpacing(0)
        for label, slot in (("Settings", self._open_settings), ("Help", self._open_help)):
            btn = QPushButton(label)
            btn.setObjectName("titlebar_btn")
            btn.clicked.connect(slot)
            tb_l.addWidget(btn)
        tb_l.addStretch()
        lbl = QLabel("OpenMFC  ·  Open Multi-Frame Compounding")
        lbl.setStyleSheet("color: #404040; font-size: 9pt; letter-spacing: 3px; background: transparent;")
        tb_l.addWidget(lbl)
        tb_l.addStretch()
        root.addWidget(titlebar)

        # Body
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Left
        self.left_panel = LeftPanel()
        self.left_panel.btn_change_output.clicked.connect(self._pick_output_folder)
        self.left_panel.btn_open_folder.clicked.connect(self._pick_input_folder)
        self.left_panel.image_selected.connect(self._load_image_to_viewfinder)
        body.addWidget(self.left_panel)

        # Center
        center = QWidget()
        center.setStyleSheet("background: #1a1a1a;")
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(0, 0, 0, 0)
        center_l.setSpacing(0)

        self.viewfinder = ViewfinderWidget()
        center_l.addWidget(self.viewfinder, 1)

        # Progress bar — sits between viewfinder and log strip
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: #1a1a1a;
                border: none;
                border-radius: 0;
            }
            QProgressBar::chunk {
                background: #7ab8f5;
                border-radius: 0;
            }
        """)
        center_l.addWidget(self.progress_bar)

        # Bottom strip
        bottom = QWidget()
        bottom.setFixedHeight(90)
        bottom.setStyleSheet("background: #161616; border-top: 1px solid #3a3a3a;")
        bot_l = QHBoxLayout(bottom)
        bot_l.setContentsMargins(0, 0, 0, 0)
        bot_l.setSpacing(0)

        # Log
        log_wrap = QWidget()
        log_wrap.setStyleSheet("background: #0e0e0e;")
        log_inner = QVBoxLayout(log_wrap)
        log_inner.setContentsMargins(0, 0, 0, 0)
        log_inner.setSpacing(0)
        log_hdr = QLabel("  LOG")
        log_hdr.setFixedHeight(20)
        log_hdr.setStyleSheet(
            "color: #444440; font-size: 7pt; letter-spacing: 2px; "
            "background: #1a1a1a; padding: 2px 6px; border-bottom: 1px solid #2a2a2a;"
        )
        log_inner.addWidget(log_hdr)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            "background: #0e0e0e; border: none; color: #666662; "
            "font-size: 8pt; padding: 4px 8px;"
        )
        log_inner.addWidget(self.log_box)
        bot_l.addWidget(log_wrap, 1)

        div = QFrame(); div.setFrameShape(QFrame.Shape.VLine)
        div.setStyleSheet("color: #3a3a3a;")
        bot_l.addWidget(div)

        # Capture buttons
        cap = QWidget(); cap.setFixedWidth(130)
        cap.setStyleSheet("background: #1e1e1e;")
        cap_l = QVBoxLayout(cap)
        cap_l.setContentsMargins(8, 8, 8, 8)
        cap_l.setSpacing(6)

        live_row = QHBoxLayout()
        live_row.addWidget(_muted("Live"))
        self.btn_live = QPushButton("Start")
        self.btn_live.setObjectName("btn_start_live")
        self.btn_live.setEnabled(False)
        self.btn_live.clicked.connect(self._on_live_capture)
        live_row.addWidget(self.btn_live)
        cap_l.addLayout(live_row)
        cap_l.addWidget(_h_rule())

        mfnr_row = QHBoxLayout()
        mfnr_row.addWidget(_muted("MFNR"))
        self.btn_mfnr = QPushButton("Start")
        self.btn_mfnr.setObjectName("btn_start_mfnr")
        self.btn_mfnr.clicked.connect(self._on_mfnr_start)
        mfnr_row.addWidget(self.btn_mfnr)
        cap_l.addLayout(mfnr_row)
        cap_l.addStretch()

        bot_l.addWidget(cap)
        center_l.addWidget(bottom)
        body.addWidget(center, 1)

        # Right
        self.options = OptionsPanel(self._camera_labels)
        body.addWidget(self.options)

        root.addLayout(body, 1)

    # ─────────────────────────────────────
    #  Signal wiring
    # ─────────────────────────────────────

    def _connect_signals(self):
        o = self.options
        o.ecc_changed.connect(lambda v: setattr(config, "ECC_THRESHOLD", round(v, 1)))
        o.hdr_mode_changed.connect(self._on_hdr_mode)
        o.ev_bracket_changed.connect(lambda v: setattr(config, "HDR_EV_BRACKET", v))
        o.fusion_changed.connect(lambda v: setattr(config, "HDR_FUSION_WEIGHT", round(v, 1)))
        o.enhance_changed.connect(lambda v: setattr(config, "ENABLE_ONNX", v.lower() == "enable"))
        o.burst_changed.connect(lambda v: setattr(config, "BURST_COUNT", max(1, v)))
        o.aspect_changed.connect(self._on_aspect_changed)
        o.camera_changed.connect(lambda v: setattr(config, "CAPTURE_DEVICE", get_index_from_label(v)))
        o.connect_camera.connect(self._on_connect_camera)
        o.open_output.connect(self._on_open_output_file)

    # ─────────────────────────────────────
    #  Logging
    # ─────────────────────────────────────

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")

        # Color-code by prefix
        if any(k in message for k in ("[pipeline]", "[mfnr]", "[hdr]", "[enhance]")):
            color = "#7ab8f5"   # blue — pipeline stages
        elif "[camera]" in message or "[capture]" in message or "[preview]" in message:
            color = "#9a9a96"   # grey — camera
        elif "[output]" in message or "[settings]" in message:
            color = "#a0c878"   # green — output/settings
        elif "error" in message.lower() or "abort" in message.lower() or "fail" in message.lower():
            color = "#e07070"   # red — errors
        elif "warn" in message.lower() or "fallback" in message.lower() or "disabled" in message.lower():
            color = "#c8a850"   # amber — warnings
        elif "ready" in message.lower():
            color = "#7ab88a"   # green — ready
        else:
            color = "#666662"   # default muted

        line = f'<span style="color:#444440">[{ts}]</span> <span style="color:{color}">{message}</span>'
        self.log_box.append(line)

        # Autoscroll to bottom
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ─────────────────────────────────────
    #  Frame handling
    # ─────────────────────────────────────

    def _on_frame_ready(self, frame: np.ndarray):
        self.viewfinder.set_frame(frame)

    def _frame_from_thread(self, frame: np.ndarray):
        """Background thread — buffer only, no Qt calls here."""
        with QMutexLocker(self._frame_mutex):
            self._pending_frame = frame.copy()
            self._frame_dirty   = True

    def _flush_pending_frame(self):
        if not self._preview_running:
            return
        with QMutexLocker(self._frame_mutex):
            if not self._frame_dirty or self._pending_frame is None:
                return
            frame = self._pending_frame
            self._frame_dirty = False
        self.viewfinder.set_frame(frame)

    def show_image_in_viewfinder(self, img: np.ndarray):
        """Safe to call from any thread."""
        self._signaler.frame_ready.emit(img)

    # ─────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────

    def _on_connect_camera(self):
        if self._preview_running:
            self._stop_preview()
            self.options.btn_connect.setText("Connect Camera")
        else:
            self.options.btn_connect.setText("Connecting…")
            self.options.btn_connect.setEnabled(False)
            threading.Thread(target=self._start_preview, daemon=True).start()

    def _start_preview(self):
        self._preview_running = True
        self._live_active     = True
        self.btn_live.setEnabled(True)

        def _set_running(val):
            self._preview_running = val
            if not val:
                self._live_active = False
                self.btn_live.setEnabled(False)
                self.options.btn_connect.setText("Connect Camera")
                self.options.btn_connect.setEnabled(True)

        threading.Thread(
            target=preview_loop,
            args=(lambda: self._preview_running, self._frame_from_thread, self.log, _set_running),
            daemon=True,
        ).start()
        self.log("[camera] Live preview started.")
        self.options.btn_connect.setText("Disconnect")
        self.options.btn_connect.setEnabled(True)

    def _stop_preview(self):
        self._preview_running = False
        self._live_active     = False
        self.btn_live.setEnabled(False)
        with QMutexLocker(self._frame_mutex):
            self._pending_frame = None
            self._frame_dirty   = False
        self._signaler.clear_view.emit()
        self.log("[camera] Live preview stopped.")

    # ─────────────────────────────────────
    #  Background camera enumeration
    # ─────────────────────────────────────

    def _enumerate_cameras_bg(self):
        """Runs on a daemon thread — never touches Qt widgets directly."""
        from devices import enumerate_cameras, get_camera_labels
        import config as _config
        self.log("[camera] Detecting cameras...")
        cameras = enumerate_cameras()
        labels  = get_camera_labels()
        if cameras:
            _config.CAPTURE_DEVICE = cameras[0]["index"]
        self._signaler.cameras_ready.emit(labels)

    def _on_cameras_ready(self, labels: list):
        """Runs on main thread via signal — safe to update widgets."""
        cmb = self.options.cmb_camera
        cmb.blockSignals(True)
        cmb.clear()
        if labels:
            cmb.addItems(labels)
            self.log(f"[camera] {len(labels)} camera(s) detected.")
        else:
            cmb.addItem("No device found")
            self.log("[camera] No cameras detected.")
        cmb.blockSignals(False)

    # ─────────────────────────────────────
    #  Pipeline
    # ─────────────────────────────────────

    def _pipeline_done(self):
        self.left_panel.refresh_gallery()

    def _on_live_capture(self):
        if not self._live_active:
            self.log("[capture] No live view — connect camera first.")
            return
        paths = capture_burst(self.log)
        run_pipeline_thread(paths, self._onnx_session, self.log,
                            self._pipeline_done, self.show_image_in_viewfinder,
                            self._signaler.progress_update.emit)

    def _on_mfnr_start(self):
        if not self._input_paths:
            self.log("[input] No images loaded.")
            return
        paths = self._input_paths[: config.BURST_COUNT]
        self.log(f"[input] Running pipeline on {len(paths)} image(s)…")
        run_pipeline_thread(paths, self._onnx_session, self.log,
                            self._pipeline_done, self.show_image_in_viewfinder,
                            self._signaler.progress_update.emit)

    # ─────────────────────────────────────
    #  File pickers
    # ─────────────────────────────────────

    def _pick_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            config.OUTPUT_PATH = d
            self.left_panel.lbl_output_path.setText(d)
            self.log(f"[output] Folder set: {d}")

    def _pick_input_folder(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Burst Images", "",
            "Images (*.jpg *.jpeg *.png *.tiff *.tif *.webp *.dng);;All Files (*.*)",
        )
        if files:
            self._input_paths = list(files)
            self.left_panel.lbl_drop.setText(f"{len(files)} file(s) loaded")
            self.log(f"[input] Loaded {len(files)} file(s).")
            img = cv2.imread(files[0])
            if img is not None:
                self.show_image_in_viewfinder(img)
        else:
            self.log("[input] No files selected.")

    def _load_image_to_viewfinder(self, path: str):
        img = cv2.imread(path)
        if img is not None:
            self.show_image_in_viewfinder(img)
            self.log(f"[preview] Loaded: {Path(path).name}")
        else:
            self.log(f"[preview] Could not read: {path}")

    def _on_open_output_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Output Image", config.OUTPUT_PATH,
            "Images (*.jpg *.jpeg *.png *.tiff *.tif *.webp *.dng);;All Files (*.*)",
        )
        if f:
            img = cv2.imread(f)
            if img is not None:
                self.show_image_in_viewfinder(img)
                self.log(f"[preview] Opened: {Path(f).name}")
            else:
                self.log(f"[preview] Could not read: {f}")

    # ─────────────────────────────────────
    #  Options callbacks
    # ─────────────────────────────────────

    def _on_hdr_mode(self, mode: str):
        config.HDR_MODE   = mode.lower()
        config.ENABLE_HDR = mode.lower() == "enable"
        self.log(f"[options] HDR: {mode}")

    def _on_aspect_changed(self, ratio: str):
        self.viewfinder.set_aspect(ratio)
        self.log(f"[viewfinder] Aspect ratio: {ratio}")

    # ─────────────────────────────────────
    #  Dialogs
    # ─────────────────────────────────────

    def _open_settings(self):
        dlg = SettingsDialog(self._camera_labels, self)
        if dlg.exec():
            self.left_panel.lbl_output_path.setText(config.OUTPUT_PATH)
            self.left_panel.refresh_gallery()
            self.log("[settings] Settings saved.")

    def _open_help(self):
        HelpDialog(self).exec()


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────

def build_ui(onnx_session=None, camera_labels: list[str] = None) -> None:
    import sys
    # Fix blurry/clipped UI on Windows high-DPI screens
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OpenMFC")
    app.setOrganizationName("MoFi Co.")

    win = MainWindow(onnx_session=onnx_session, camera_labels=camera_labels)

    import os
    for ico in (os.path.join("assets", "icon.ico"), os.path.join("assets", "OpenMFC ICON.png")):
        if os.path.exists(ico):
            win.setWindowIcon(QIcon(ico))
            break

    win.show()
    win.log("OpenMFC Beta 0.01.0 ready.")
    if not camera_labels:
        win.log("[camera] No cameras detected.")
    if not onnx_session:
        win.log("[onnx] No model loaded — Enhance disabled.")

    sys.exit(app.exec())