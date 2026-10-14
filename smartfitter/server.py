"""Local HTTP server: serves the interface and a small JSON API (127.0.0.1 only)."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import core

if getattr(sys, "frozen", False):  # PyInstaller build: data files live under sys._MEIPASS
    _BUNDLE = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    STATIC = _BUNDLE / "smartfitter" / "static"
    PLOTLY_JS = _BUNDLE / "plotly" / "package_data" / "plotly.min.js"
else:
    STATIC = Path(__file__).with_name("static")
    PLOTLY_JS = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"


class Api:
    """Every public method is callable from the interface as POST /api/<name>."""

    def __init__(self) -> None:
        self.ws = core.Workspace()
        self.window = None          # pywebview window when running as a desktop app
        self.pending_open: list[str] = []
        self.session_lock = threading.Lock()

    # -- app / settings ---------------------------------------------------
    def app_info(self) -> dict:
        return {"version": core.APP_VERSION, "settings": core.load_settings(), "pending_open": self.take_pending(),
                "desktop": self.window is not None, "home": str(Path.home())}

    def queue_open(self, paths: list[str]) -> bool:
        """Called by a second launch (e.g. double-clicking a .mat file) to open files in this window."""
        self.pending_open.extend(paths)
        if self.window is not None:
            try:
                self.window.restore()
                self.window.show()
            except Exception:
                pass
        return True

    def take_pending(self) -> list[str]:
        items, self.pending_open = self.pending_open, []
        return items

    def settings_set(self, patch: dict) -> dict:
        return core.save_settings(patch)

    def remember(self, kind: str, path: str) -> dict:
        core.remember(kind, path)
        return core.load_settings()

    # -- files ------------------------------------------------------------
    def list_dir(self, path: str) -> dict:
        return core.list_dir(path)

    def peek(self, paths: list[str]) -> list[dict]:
        return core.clean(self.ws.peek_many(paths))

    def stat(self, paths: list[str]) -> dict:
        out = {}
        for p in paths:
            try:
                out[p] = os.stat(p).st_mtime
            except OSError:
                out[p] = None
        return out

    def load(self, path: str, detector: str | None = None, mode: str = "contrast", proc: dict | None = None) -> dict:
        return core.trace_payload(self.ws, path, detector, mode, proc)

    def load_all_detectors(self, path: str, mode: str = "contrast", proc: dict | None = None) -> dict:
        ds = self.ws.dataset(path, mode)
        return {det: core.trace_payload(self.ws, path, det, mode, proc) for det in ds.detectors}

    def save_upload(self, name: str, data: str) -> str:
        """Fallback for drag-and-drop when no real path is available. Each distinct content gets its own subfolder, so
        two different files with the same name (e.g. from two dated folders) never overwrite each other."""
        import hashlib

        raw = base64.b64decode(data.split(",", 1)[-1])
        folder = Path(tempfile.gettempdir()) / "SmartFitterDrops" / hashlib.sha1(raw).hexdigest()[:12]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / Path(name).name
        if not target.exists():
            target.write_bytes(raw)
        return str(target)

    # -- desktop drag & drop: real paths -----------------------------------
    # pywebview only reveals full paths to Python-side DOM drop listeners (__main__ registers one); the page asks here.
    def record_drop(self, paths: list[str]) -> None:
        import time

        with self.session_lock:
            self._drops = [(time.monotonic(), p) for p in paths if p]

    def drop_paths(self, names: list[str], wait_ms: int = 700) -> list[str | None]:
        """Real paths for just-dropped files, matched by name (waits briefly for the Python-side listener)."""
        import time

        deadline = time.monotonic() + wait_ms / 1000
        while True:
            with self.session_lock:
                recent = [p for t, p in getattr(self, "_drops", []) if time.monotonic() - t < 10]
            found = [next((p for p in recent if Path(p).name == n), None) for n in names]
            if all(found) or time.monotonic() >= deadline:
                return found
            time.sleep(0.05)

    # -- analysis ---------------------------------------------------------
    def fit(self, path: str, detector: str | None, mode: str, proc: dict, spec: dict) -> dict:
        return core.run_fit(self.ws, path, detector, mode, proc or {}, spec or {})

    def spots(self, path: str, detector: str, mode: str, opts: dict) -> dict:
        return core.spots(self.ws, path, detector, mode, opts or {})

    def compare_spots(self, a: dict, b: dict, mode: str = "contrast") -> dict:
        return core.compare_spots(self.ws, a, b, mode)

    def map_overlay(self, first: dict, second: dict, opts: dict) -> dict:
        return core.map_overlay(self.ws, first, second, opts or {})

    def time_stats(self, path: str, mode: str, windows: dict) -> dict:
        return core.time_stats(self.ws, path, mode, windows or {})

    def periodogram(self, path: str, detector: str, mode: str) -> dict:
        return core.periodogram(self.ws, path, detector, mode)

    # -- batch ------------------------------------------------------------
    def batch_start(self, opts: dict) -> str:
        return core.batch_start(self.ws, opts)

    def batch_status(self, job: str) -> dict:
        j = self.ws.jobs.get(job)
        if not j:
            raise ValueError("Unknown batch job")
        return {k: v for k, v in j.items() if k != "cancel"}

    def batch_cancel(self, job: str) -> bool:
        if job in self.ws.jobs:
            self.ws.jobs[job]["cancel"] = True
        return True

    def batch_cli(self, opts: dict) -> str:
        return core.batch_cli(opts)

    def batch_summary(self, output_dir: str) -> list[dict]:
        return core.read_batch_summary(output_dir)

    # -- export / sessions ------------------------------------------------
    def export(self, req: dict) -> dict:
        return core.export_results(self.ws, req)

    def write_text(self, path: str, text: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        return path

    def write_data_url(self, path: str, data: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(data.split(",", 1)[-1]))
        return path

    def save_session(self, path: str, doc: dict) -> str:
        from nvfit.sessions import source_descriptor

        doc = dict(doc)
        doc["format"] = "smartfitter-session"
        doc["version"] = 1
        doc["sources"] = {}
        physical_sources = [src for src in doc.get("files", []) if not src.startswith("combined:")]
        for saved in doc.get("state", []):
            physical_sources.extend(r["path"] for r in (saved.get("combinedRecipe") or {}).get("runs", []))
        for src in dict.fromkeys(physical_sources):
            try:
                doc["sources"][src] = source_descriptor(src, path)
            except OSError:
                pass
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(core.clean(doc), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        core.remember("recent_files", path)
        return path

    def load_session(self, path: str) -> dict:
        from nvfit.sessions import resolve_source

        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("format") != "smartfitter-session":
            raise ValueError("Not a SmartFitter v2 session file.")
        moved, changed = {}, []
        for src, desc in (doc.get("sources") or {}).items():
            found, was_changed = resolve_source(desc, path)
            if found and str(found) != src:
                moved[src] = str(found)
            if was_changed:
                changed.append(src)
        doc["moved"], doc["changed"] = moved, changed
        return doc

    # -- desktop integration ----------------------------------------------
    def dialog(self, kind: str, title: str = "", directory: str = "", filename: str = "", filters: list | None = None) -> Any:
        """kind: open_files | open_folder | save. Returns a path, a list of paths, or None."""
        directory = directory or core.load_settings().get("last_dir") or str(Path.home())
        if self.window is not None:
            import webview

            types = {"open_files": webview.FileDialog.OPEN, "open_folder": webview.FileDialog.FOLDER, "save": webview.FileDialog.SAVE}[kind]
            ft = tuple(filters or (["MATLAB data (*.mat)", "All files (*.*)"] if kind == "open_files" else []))
            result = self.window.create_file_dialog(types, directory=directory, allow_multiple=kind == "open_files",
                                                    save_filename=filename, file_types=ft)
            if not result:
                return None
            if kind == "save":
                return result if isinstance(result, str) else result[0]
            return list(result) if kind == "open_files" else result[0]
        return _tk_dialog(kind, title, directory, filename, filters)

    def reveal(self, path: str) -> bool:
        p = Path(path)
        if p.is_dir():
            os.startfile(str(p))  # noqa: S606 - local desktop convenience
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])
        return True

    def open_path(self, path: str) -> bool:
        os.startfile(path)  # noqa: S606
        return True

    def clipboard_image(self, data: str) -> bool:
        """Put a PNG on the Windows clipboard (used when the browser clipboard is unavailable)."""
        tmp = Path(tempfile.gettempdir()) / "smartfitter_clip.png"
        tmp.write_bytes(base64.b64decode(data.split(",", 1)[-1]))
        script = ("Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
                  f"$img=[System.Drawing.Image]::FromFile('{tmp}'); [System.Windows.Forms.Clipboard]::SetImage($img); $img.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], check=True, capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True

    def clipboard_text(self, text: str) -> bool:
        subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", "$input | Set-Clipboard"], input=text, text=True,
                       check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True


def _tk_dialog(kind, title, directory, filename, filters):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "open_files":
            res = filedialog.askopenfilenames(title=title or "Open", initialdir=directory,
                                              filetypes=[("MATLAB data", "*.mat"), ("All files", "*.*")])
            return list(res) or None
        if kind == "open_folder":
            return filedialog.askdirectory(title=title or "Choose folder", initialdir=directory) or None
        return filedialog.asksaveasfilename(title=title or "Save as", initialdir=directory, initialfile=filename) or None
    finally:
        root.destroy()


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    methods: dict[str, Callable] = {name: getattr(api, name) for name in dir(api) if not name.startswith("_") and callable(getattr(api, name))}
    # Feature modules (smartfitter/ext_*.py) add endpoints via ENDPOINTS = {name: fn(ws, **kwargs)}.
    import importlib
    import pkgutil
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if info.name.startswith("ext_"):
            module = importlib.import_module(f"{__package__}.{info.name}")
            for name, fn in getattr(module, "ENDPOINTS", {}).items():
                methods[name] = (lambda f: lambda **kw: f(api.ws, **kw))(fn)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", ""):
                path = "/index.html"
            if path == "/vendor/plotly.min.js":
                target = PLOTLY_JS
            else:
                target = (STATIC / path.lstrip("/")).resolve()
                if STATIC.resolve() not in target.parents:
                    return self._send(404, b"not found", "text/plain")
            if not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix == ".js":
                ctype = "text/javascript"
            self._send(200, target.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))

        def do_POST(self):
            if not self.path.startswith("/api/"):
                return self._send(404, b"not found", "text/plain")
            name = self.path[5:].split("?", 1)[0]
            fn = methods.get(name)
            length = int(self.headers.get("Content-Length") or 0)
            try:
                args = json.loads(self.rfile.read(length) or b"{}") if length else {}
                if fn is None:
                    raise ValueError(f"Unknown API call: {name}")
                result = fn(**args)
                body = json.dumps({"ok": True, "result": core.clean(result)}, allow_nan=False).encode()
            except Exception as exc:  # report to the interface instead of crashing the server
                body = json.dumps({"ok": False, "error": str(exc) or exc.__class__.__name__,
                                   "trace": traceback.format_exc()[-3000:]}).encode()
            self._send(200, body, "application/json")

    return Handler


def serve(api: Api, port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(api))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
