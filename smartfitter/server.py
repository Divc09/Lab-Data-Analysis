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
import secrets
import hmac
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


MAX_REQUEST_BYTES = 64 * 1024 * 1024


class Api:
    """Every public method is callable from the interface as POST /api/<name>."""

    def __init__(self, *, max_request_bytes: int = MAX_REQUEST_BYTES) -> None:
        if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes < 1:
            raise ValueError("max_request_bytes must be a positive integer.")
        self.max_request_bytes = max_request_bytes
        self.ws = core.Workspace()
        self.window = None  # pywebview window when running as a desktop app
        self.pending_open: list[str] = []
        self.session_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)

    @core.plain_endpoint
    def ping(self) -> dict:
        return {"version": core.APP_VERSION}

    @core.plain_endpoint
    def validate_analysis(self, analysis: dict) -> dict:
        return core.validate_analysis(self.ws, analysis)

    # -- app / settings ---------------------------------------------------
    @core.plain_endpoint
    def app_info(self) -> dict:
        return {
            "version": core.APP_VERSION,
            "settings": core.load_settings(),
            "pending_open": self.take_pending(),
            "desktop": self.window is not None,
            "home": str(Path.home()),
        }

    @core.plain_endpoint
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

    @core.plain_endpoint
    def take_pending(self) -> list[str]:
        items, self.pending_open = self.pending_open, []
        return items

    @core.plain_endpoint
    def settings_set(self, patch: dict) -> dict:
        return core.save_settings(patch)

    @core.plain_endpoint
    def remember(self, kind: str, path: str) -> dict:
        core.remember(kind, path)
        return core.load_settings()

    # -- files ------------------------------------------------------------
    @core.plain_endpoint
    def list_dir(self, path: str) -> dict:
        return core.list_dir(path)

    @core.plain_endpoint
    def peek(self, paths: list[str]) -> list[dict]:
        return core.clean(self.ws.peek_many(paths))

    @core.plain_endpoint
    def stat(self, paths: list[str]) -> dict:
        out = {}
        for p in paths:
            try:
                out[p] = os.stat(p).st_mtime
            except OSError:
                out[p] = None
        return out

    @core.plain_endpoint
    def load_info(self, path: str, mode: str = "contrast") -> dict:
        return core.load_info(self.ws, path, mode)

    @core.owned_result
    def load(self, path: str, detector: str | None = None, mode: str = "contrast", proc: dict | None = None) -> dict:
        return core.trace_payload(self.ws, path, detector, mode, proc)

    @core.plain_endpoint
    def load_all_detectors(self, path: str, mode: str = "contrast", proc: dict | None = None) -> dict:
        ds = self.ws.dataset(path, mode)
        return {det: core.trace_payload(self.ws, path, det, mode, proc) for det in ds.detectors}

    @core.plain_endpoint
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
    @core.plain_endpoint
    def record_drop(self, paths: list[str]) -> None:
        import time

        with self.session_lock:
            self._drops = [(time.monotonic(), p) for p in paths if p]

    @core.plain_endpoint
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
    @core.owned_result
    def fit(self, path: str, detector: str | None, mode: str, proc: dict, spec: dict) -> dict:
        return core.run_fit(self.ws, path, detector, mode, proc or {}, spec or {})

    @core.owned_result
    def spots(self, path: str, detector: str, mode: str, opts: dict) -> dict:
        return core.spots(self.ws, path, detector, mode, opts or {})

    @core.plain_endpoint
    def compare_spots(self, a: dict, b: dict, mode: str = "contrast") -> dict:
        return core.compare_spots(self.ws, a, b, mode)

    @core.owned_result
    def map_overlay(self, first: dict, second: dict, opts: dict) -> dict:
        return core.map_overlay(self.ws, first, second, opts or {})

    @core.owned_result
    def time_stats(self, path: str, mode: str, windows: dict) -> dict:
        return core.time_stats(self.ws, path, mode, windows or {})

    @core.plain_endpoint
    def periodogram(self, path: str, detector: str, mode: str) -> dict:
        return core.periodogram(self.ws, path, detector, mode)

    # -- batch ------------------------------------------------------------
    @core.plain_endpoint
    def batch_start(self, opts: dict) -> str:
        return core.batch_start(self.ws, opts)

    @core.plain_endpoint
    def batch_import(self, manifest_path: str, row_id: str) -> dict:
        return core.batch_import(self.ws, manifest_path, row_id)

    @core.plain_endpoint
    def batch_rerun(self, manifest_path: str, rows: list[str]) -> str:
        return core.batch_rerun(self.ws, manifest_path, rows)

    @core.plain_endpoint
    def batch_resume(self, manifest_path: str) -> str:
        return core.batch_resume(self.ws, manifest_path)

    @core.plain_endpoint
    def batch_run_record(self, output_dir: str):
        return core.batch_run_record(output_dir)

    @core.plain_endpoint
    def batch_status(self, job: str) -> dict:
        j = self.ws.jobs.get(job)
        if not j:
            raise ValueError("Unknown batch job")
        return {k: v for k, v in j.items() if k != "cancel"}

    @core.plain_endpoint
    def batch_cancel(self, job: str) -> bool:
        if job in self.ws.jobs:
            self.ws.jobs[job]["cancel"] = True
        return True

    @core.plain_endpoint
    def batch_cli(self, opts: dict) -> str:
        return core.batch_cli(opts)

    @core.plain_endpoint
    def batch_summary(self, output_dir: str) -> list[dict]:
        return core.read_batch_summary(output_dir)

    # -- export / sessions ------------------------------------------------
    @core.plain_endpoint
    def export(self, req: dict) -> dict:
        return core.export_results(self.ws, req)

    @core.plain_endpoint
    def write_text(self, path: str, text: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        return path

    @core.plain_endpoint
    def write_data_url(self, path: str, data: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(data.split(",", 1)[-1]))
        return path

    @core.plain_endpoint
    def save_session(self, path: str, doc: dict) -> str:
        from nvfit.sessions import source_descriptor

        doc = dict(doc)
        doc["format"] = "smartfitter-session"
        doc["version"] = 2
        doc["sources"] = {}
        physical_sources = [src for src in doc.get("files", []) if not src.startswith("combined:")]
        for saved in doc.get("state", []):
            physical_sources.extend(r["path"] for r in (saved.get("combinedRecipe") or {}).get("runs", []))

        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if (
                        key in ("path", "source", "refPath", "ref_path")
                        and isinstance(item, str)
                        and item.lower().endswith(".mat")
                    ):
                        physical_sources.append(item)
                    if key in ("dependencies", "sources") and isinstance(item, dict):
                        physical_sources.extend(p for p in item if str(p).lower().endswith(".mat"))
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.lower().endswith(".mat"):
                        physical_sources.append(item)
                    else:
                        collect(item)

        collect(doc)
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

    @core.plain_endpoint
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
    @core.plain_endpoint
    def dialog(
        self, kind: str, title: str = "", directory: str = "", filename: str = "", filters: list | None = None
    ) -> Any:
        """kind: open_files | open_folder | save. Returns a path, a list of paths, or None."""
        directory = directory or core.load_settings().get("last_dir") or str(Path.home())
        if self.window is not None:
            import webview

            types = {
                "open_files": webview.FileDialog.OPEN,
                "open_folder": webview.FileDialog.FOLDER,
                "save": webview.FileDialog.SAVE,
            }[kind]
            ft = tuple(filters or (["MATLAB data (*.mat)", "All files (*.*)"] if kind == "open_files" else []))
            result = self.window.create_file_dialog(
                types, directory=directory, allow_multiple=kind == "open_files", save_filename=filename, file_types=ft
            )
            if not result:
                return None
            if kind == "save":
                return result if isinstance(result, str) else result[0]
            return list(result) if kind == "open_files" else result[0]
        return _tk_dialog(kind, title, directory, filename, filters)

    @core.plain_endpoint
    def reveal(self, path: str) -> bool:
        p = Path(path)
        if p.is_dir():
            os.startfile(str(p))  # noqa: S606 - local desktop convenience
        else:
            subprocess.Popen(["explorer", "/select,", str(p)])
        return True

    @core.plain_endpoint
    def open_path(self, path: str) -> bool:
        os.startfile(path)  # noqa: S606
        return True

    @core.plain_endpoint
    def clipboard_image(self, data: str) -> bool:
        """Put a PNG on the Windows clipboard (used when the browser clipboard is unavailable)."""
        tmp = Path(tempfile.gettempdir()) / "smartfitter_clip.png"
        tmp.write_bytes(base64.b64decode(data.split(",", 1)[-1]))
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; "
            f"$img=[System.Drawing.Image]::FromFile('{tmp}'); [System.Windows.Forms.Clipboard]::SetImage($img); $img.Dispose()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", script],
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True

    @core.plain_endpoint
    def clipboard_text(self, text: str) -> bool:
        subprocess.run(
            ["powershell", "-NoProfile", "-STA", "-Command", "$input | Set-Clipboard"],
            input=text,
            text=True,
            check=True,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return True


def _tk_dialog(kind, title, directory, filename, filters):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "open_files":
            res = filedialog.askopenfilenames(
                title=title or "Open", initialdir=directory, filetypes=[("MATLAB data", "*.mat"), ("All files", "*.*")]
            )
            return list(res) or None
        if kind == "open_folder":
            return filedialog.askdirectory(title=title or "Choose folder", initialdir=directory) or None
        return (
            filedialog.asksaveasfilename(title=title or "Save as", initialdir=directory, initialfile=filename) or None
        )
    finally:
        root.destroy()


def make_handler(api: Api) -> type[BaseHTTPRequestHandler]:
    methods: dict[str, Callable] = {
        name: core.bind_endpoint(getattr(api, name))
        for name in dir(api)
        if not name.startswith("_") and callable(getattr(api, name))
    }
    # Feature modules (smartfitter/ext_*.py) add endpoints via ENDPOINTS = {name: fn(ws, **kwargs)}.
    import importlib
    import pkgutil

    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if info.name.startswith("ext_"):
            module = importlib.import_module(f"{__package__}.{info.name}")
            for name, fn in getattr(module, "ENDPOINTS", {}).items():
                methods[name] = core.bind_endpoint(fn, api.ws)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, etag: str | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if etag:
                # Static files: kept by the browser but revalidated on every load, so edits show at once while
                # unchanged files (Plotly is 4.7 MB) come back as 304 and reuse the browser's compiled-code cache.
                self.send_header("Cache-Control", "no-cache")
                self.send_header("ETag", etag)
            else:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            if code >= 400:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._local_host():
                return self._send(403, b"Forbidden Host", "text/plain")
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
            ctype += "; charset=utf-8" if ctype.startswith("text") else ""
            stat = target.stat()
            etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                return self._send(304, b"", ctype, etag)
            self._send(200, target.read_bytes(), ctype, etag)

        def _reject(self, code, message):
            body = json.dumps({"ok": False, "error": message}).encode()
            self._send(code, body, "application/json")
            return None

        def _request_length(self):
            """Reject invalid/oversized framing before reading or dispatching a body."""
            if not self._local_host():
                return self._reject(403, "Forbidden Host")
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            if origin and origin not in allowed:
                return self._reject(403, "Forbidden Origin")
            if not hmac.compare_digest(self.headers.get("X-SmartFitter-Token", ""), api.token):
                return self._reject(403, "Session token missing or invalid. Open SmartFitter from its launcher.")
            if self.headers.get("Transfer-Encoding"):
                return self._reject(400, "Transfer-Encoding is not supported; send a bounded JSON body.")
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                return self._reject(411, "Content-Length is required.")
            if (
                len(lengths) != 1
                or not lengths[0].strip().isascii()
                or not lengths[0].strip().isdigit()
                or len(lengths[0].strip()) > 20
            ):
                return self._reject(400, "Invalid Content-Length.")
            length = int(lengths[0])
            if length > api.max_request_bytes:
                return self._reject(413, f"Request body exceeds the {api.max_request_bytes}-byte limit.")
            if self.headers.get_content_type() != "application/json":
                return self._reject(415, "Content-Type must be application/json.")
            return length

        def handle_expect_100(self):
            if self._request_length() is None:
                return False
            self.send_response_only(100)
            self.end_headers()
            return True

        def do_POST(self):
            length = self._request_length()
            if length is None:
                return
            if not self.path.startswith("/api/"):
                return self._send(404, b"not found", "text/plain")
            name = self.path[5:].split("?", 1)[0]
            fn = methods.get(name)
            try:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    return self._reject(400, "Incomplete request body.")
                args = json.loads(payload or b"{}")
                if not isinstance(args, dict):
                    return self._reject(400, "The JSON body must be an object.")
            except (ValueError, UnicodeError):
                return self._reject(400, "Invalid JSON request body.")
            try:
                if fn is None:
                    raise ValueError(f"Unknown API call: {name}")
                result = fn(**args)
                body = json.dumps({"ok": True, "result": core.clean(result)}, allow_nan=False).encode()
            except Exception as exc:  # report to the interface instead of crashing the server
                body = json.dumps(
                    {"ok": False, "error": str(exc) or exc.__class__.__name__, "trace": traceback.format_exc()[-3000:]}
                ).encode()
            self._send(200, body, "application/json")

        def _local_host(self):
            return self.headers.get("Host", "") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

    return Handler


def serve(api: Api, port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(api))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
