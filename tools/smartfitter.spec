# PyInstaller spec: onedir, windowed SmartFitter (no PySide6). Run via tools/build_portable.ps1.
from pathlib import Path
import plotly
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).parent
datas, binaries, hiddenimports = [], [], []

for pkg in ("webview", "clr_loader", "pythonnet"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

# Only the plotly.min.js bundle is needed (served at /vendor/plotly.min.js), not the plotly python package.
datas += [(str(Path(plotly.__file__).parent / "package_data" / "plotly.min.js"), "plotly/package_data")]
datas += [(str(ROOT / "smartfitter" / "static"), "smartfitter/static")]
# ext_*.py endpoint modules are discovered dynamically, so list every submodule explicitly.
hiddenimports += collect_submodules("smartfitter") + collect_submodules("nvfit.models")
hiddenimports += ["clr", "matplotlib.backends.backend_agg"]

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
