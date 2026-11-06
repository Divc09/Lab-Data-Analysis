# PyInstaller spec: onedir, windowed SmartFitter (no PySide6). Run via tools/build_portable.ps1.
import re
import sys
from pathlib import Path
import plotly
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []

IS_MAC = sys.platform == "darwin"  # macOS: SmartFitter.app via BUNDLE below (built by .github/workflows/build.yml)

# Windows needs pythonnet for WebView2; macOS uses pywebview's Cocoa backend (pyobjc, found by PyInstaller hooks).
for pkg in ("webview",) if IS_MAC else ("webview", "clr_loader", "pythonnet"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Only the plotly.min.js bundle is needed (served at /vendor/plotly.min.js), not the plotly python package.
datas += [(str(Path(plotly.__file__).parent / "package_data" / "plotly.min.js"), "plotly/package_data")]
datas += [(str(ROOT / "smartfitter" / "static"), "smartfitter/static")]
# ext_*.py endpoint modules are discovered dynamically, so list every submodule explicitly.
hiddenimports += collect_submodules("smartfitter") + collect_submodules("nvfit.models")
hiddenimports += ["matplotlib.backends.backend_agg"] + ([] if IS_MAC else ["clr"])

a = Analysis(
    [str(ROOT / "tools" / "smartfitter_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    excludes=["PySide6", "PySide2", "PyQt5", "PyQt6", "shiboken6", "plotly", "pytest",
              "IPython", "jupyter", "notebook", "matplotlib.backends.backend_qt5agg",
              "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_tkagg"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SmartFitter", console=False,
          icon=str(ROOT / "smartfitter" / "static" / "icon.ico"), upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="SmartFitter", upx=False)
if IS_MAC:
    version = re.search(r'APP_VERSION = "([^"]+)"', (ROOT / "smartfitter" / "core.py").read_text(encoding="utf-8"))[1]
    app = BUNDLE(coll, name="SmartFitter.app", icon=str(ROOT / "smartfitter" / "static" / "icon.ico"),
                 bundle_identifier="edu.backlundlab.smartfitter",
                 info_plist={"CFBundleShortVersionString": version, "CFBundleVersion": version,
                             "NSHighResolutionCapable": True})
