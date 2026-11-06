"""Start SmartFitter v2:  python -m smartfitter [files...] [--browser] [--port N]

Opens a native desktop window (pywebview / WebView2). If that is unavailable it
falls back to a chromeless Microsoft Edge app window. A second launch hands its
files to the running window instead of starting another copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import webbrowser
from pathlib import Path

from . import core
from .server import Api, serve

LOCK = core.SETTINGS_DIR / "instance.json"


def _existing_instance() -> int | None:
    try:
        record = json.loads(LOCK.read_text())
        port = int(record["port"])
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/ping",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if not json.loads(urllib.request.urlopen(request, timeout=0.7).read()).get("ok"):
            return None
        return port
    except Exception:
        return None


def _post(port: int, name: str, payload: dict) -> None:
    record = json.loads(LOCK.read_text())
    if int(record["port"]) != port:
        raise ValueError("SmartFitter instance changed; retry the launch.")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/{name}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=3).read()


def _edge() -> str | None:
    for candidate in (
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartFitter v2")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--browser", action="store_true", help="Serve only and open the default browser (development).")
    parser.add_argument("--serve", action="store_true", help="Serve only, do not open any window.")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    files = [str(Path(f).resolve()) for f in args.files if Path(f).exists()]

    if not (args.browser or args.serve):
        port = _existing_instance()
        if port:
            _post(port, "queue_open", {"paths": files})
            return

    api = Api()
    api.pending_open = files
    server = serve(api, args.port)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    core.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.serve or args.browser:
        print(f"SmartFitter serving at {url}", flush=True)
        if args.browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return

    LOCK.write_text(json.dumps({"port": port, "pid": os.getpid()}))
    try:
        import webview

        icon = Path(__file__).with_name("static") / "icon.ico"
        api.window = webview.create_window(
            "SmartFitter",
            url,
            width=1500,
            height=940,
            min_size=(1000, 640),
            background_color="#0E1014",
            text_select=True,
        )

        def bind_drop(window):
            # A Python-side listener is what makes pywebview attach real paths to dropped files; the page then asks
            # Api.drop_paths, so dropped files open from where they live instead of as temporary copies.
            try:
                from webview.dom import DOMEventHandler

                def on_drop(event):
                    files = (event.get("dataTransfer") or {}).get("files") or []
                    api.record_drop([f.get("pywebviewFullPath") for f in files])

                def attach():  # DOM listeners live in the page, so re-attach after every (re)load
                    try:
                        window.dom.document.events.drop += DOMEventHandler(
                            on_drop, prevent_default=False, stop_propagation=False
                        )
                    except Exception:
                        pass

                window.events.loaded += attach
                attach()
            except Exception:
                pass  # older pywebview: uploads remain the fallback

        # Windows: WebView2 (edgechromium). macOS/Linux: pywebview's native backend (WebKit / GTK / Qt).
        gui = "edgechromium" if sys.platform == "win32" else None
        webview.start(bind_drop, (api.window,), gui=gui, icon=str(icon) if icon.exists() else None)
    except Exception as exc:  # pywebview missing or WebView2 unavailable: use Edge in app mode
        edge = _edge()
        if not edge:
            webbrowser.open(url)
            server.serve_forever()
            return
        profile = Path(tempfile.gettempdir()) / "SmartFitterEdgeProfile"
        sys.stderr.write(f"Desktop window unavailable ({exc}); using Edge app window.\n")
        subprocess.run([edge, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1500,940", "--no-first-run"])
    finally:
        try:
            if json.loads(LOCK.read_text()).get("pid") == os.getpid():
                LOCK.unlink()
        except Exception:
            pass
        server.shutdown()


if __name__ == "__main__":
    main()
