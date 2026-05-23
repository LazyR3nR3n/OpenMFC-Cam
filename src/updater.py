import threading
import requests
import subprocess
import tempfile
import os

from PyQt6.QtWidgets import QMessageBox, QProgressDialog
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from version import __version__

GITHUB_API_URL = "https://api.github.com/repos/LazyR3nR3n/OpenMFC-Cam/releases/latest"


# --- Version comparison ---

def _is_newer(latest_tag: str, current_tag: str) -> bool:
    """
    Compare two tags of the form beta_X.Y.Z.
    Returns True if latest_tag is newer than current_tag.
    """
    def parse(tag: str):
        # Strip prefix, split into int tuple
        numeric = tag.lower().replace("beta_", "").replace("beta-", "")
        try:
            return tuple(int(x) for x in numeric.split("."))
        except ValueError:
            return (0, 0, 0)

    return parse(latest_tag) > parse(current_tag)


# --- Download thread ---

class _DownloadThread(QThread):
    progress = pyqtSignal(int)       # 0–100
    finished = pyqtSignal(str)       # temp path to installer
    error = pyqtSignal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            suffix = ".exe" if self.url.endswith(".exe") else ".bin"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        self.progress.emit(int(downloaded / total * 100))

            tmp.close()
            self.finished.emit(tmp.name)

        except Exception as e:
            self.error.emit(str(e))


# --- Public API ---

def check_for_updates(parent=None, silent: bool = True):
    """
    Check GitHub for a newer release.
    - silent=True  → only show dialog if update is available (startup behavior)
    - silent=False → always show result (manual check from menu)
    """
    def _check():
        try:
            response = requests.get(GITHUB_API_URL, timeout=5)
            response.raise_for_status()
            data = response.json()

            latest_tag = data.get("tag_name", "")
            release_name = data.get("name", latest_tag)
            release_notes = data.get("body", "No release notes provided.")

            # Find the .exe asset
            assets = data.get("assets", [])
            installer_url = None
            for asset in assets:
                if asset["name"].endswith(".exe"):
                    installer_url = asset["browser_download_url"]
                    break

            if _is_newer(latest_tag, __version__):
                _show_update_dialog(parent, release_name, latest_tag, release_notes, installer_url)
            elif not silent:
                QMessageBox.information(
                    parent,
                    "No Updates",
                    f"You're already on the latest version ({__version__})."
                )

        except Exception as e:
            if not silent:
                QMessageBox.warning(
                    parent,
                    "Update Check Failed",
                    f"Could not reach GitHub:\n{e}"
                )

    # Run network call off the main thread
    threading.Thread(target=_check, daemon=True).start()


def _show_update_dialog(parent, release_name: str, tag: str, notes: str, installer_url: str | None):
    """Show the Mihon-style update popup."""
    # Trim notes if too long
    notes_preview = notes[:500] + ("..." if len(notes) > 500 else "")

    msg = QMessageBox(parent)
    msg.setWindowTitle("Update Available")
    msg.setText(f"<b>{release_name}</b> is available.<br><br>"
                f"You have: <code>{__version__}</code><br>"
                f"Latest:&nbsp;&nbsp;&nbsp; <code>{tag}</code>")
    msg.setDetailedText(notes_preview)
    msg.setIcon(QMessageBox.Icon.Information)

    if installer_url:
        update_btn = msg.addButton("Update Now", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        if msg.clickedButton() == update_btn:
            _download_and_launch(parent, installer_url)
    else:
        # No .exe asset found — send to releases page
        msg.addButton("Open Releases Page", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        if msg.clickedButton().text() == "Open Releases Page":
            import webbrowser
            webbrowser.open("https://github.com/LazyR3nR3n/OpenMFC-Cam/releases/latest")


def _download_and_launch(parent, url: str):
    """Download installer with progress dialog, then launch it."""
    progress_dialog = QProgressDialog("Downloading update...", "Cancel", 0, 100, parent)
    progress_dialog.setWindowTitle("OpenMFC Updater")
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setValue(0)

    thread = _DownloadThread(url)

    def on_progress(val):
        progress_dialog.setValue(val)

    def on_finished(path):
        progress_dialog.close()
        # Launch installer, then exit app so it can replace the exe
        subprocess.Popen([path], creationflags=subprocess.CREATE_NO_WINDOW)
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()

    def on_error(msg):
        progress_dialog.close()
        QMessageBox.critical(parent, "Download Failed", f"Could not download update:\n{msg}")

    def on_cancel():
        thread.terminate()

    thread.progress.connect(on_progress)
    thread.finished.connect(on_finished)
    thread.error.connect(on_error)
    progress_dialog.canceled.connect(on_cancel)

    thread.start()
    progress_dialog.exec()