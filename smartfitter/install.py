"""Create SmartFitter launchers on Windows (no admin rights needed).

    .venv\\Scripts\\python.exe -m smartfitter.install            Start menu + desktop + repo shortcuts
    .venv\\Scripts\\python.exe -m smartfitter.install --associate  also add SmartFitter to "Open with" for .mat files
    .venv\\Scripts\\python.exe -m smartfitter.install --remove     remove everything this script created

Shortcuts start the app with this checkout's virtual environment (pythonw, so no
console window). "--associate" only adds SmartFitter to the Open-with list; it does
not change which program opens .mat files by default (MATLAB keeps that).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICON = Path(__file__).with_name("static") / "icon.ico"
PROGID = "SmartFitter.MatFile"
LAUNCHER = Path(__file__).with_name("launch.pyw")


def _pythonw() -> Path:
    candidate = ROOT / ".venv" / "Scripts" / "pythonw.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable).with_name("pythonw.exe")


def _shortcut_targets() -> list[Path]:
    appdata = Path(os.environ["APPDATA"])
    desktop = Path(subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                                  capture_output=True, text=True, check=True).stdout.strip())
    return [appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "SmartFitter.lnk", desktop / "SmartFitter.lnk", ROOT / "SmartFitter.lnk"]


def _make_shortcut(path: Path) -> None:
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:SF_LNK);"
        "$s.TargetPath=$env:SF_TARGET;$s.Arguments=$env:SF_ARGS;$s.WorkingDirectory=$env:SF_ROOT;"
        "$s.IconLocation=$env:SF_ICON;$s.Description='SmartFitter: NV-center data analysis';$s.Save()"
    )
    env = {**os.environ, "SF_ARGS": f'"{LAUNCHER}"', "SF_LNK": str(path), "SF_TARGET": str(_pythonw()), "SF_ROOT": str(ROOT), "SF_ICON": str(ICON)}
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=True)


def _associate(remove: bool = False) -> None:
    import winreg

    base = r"Software\Classes"
    if remove:
        for key in (rf"{base}\{PROGID}\shell\open\command", rf"{base}\{PROGID}\shell\open", rf"{base}\{PROGID}\shell",
                    rf"{base}\{PROGID}\DefaultIcon", rf"{base}\{PROGID}"):
            try:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
            except OSError:
                pass
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, rf"{base}\.mat\OpenWithProgids", 0, winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, PROGID)
        except OSError:
            pass
        return
    command = f'"{_pythonw()}" "{LAUNCHER}" "%1"'
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "MATLAB data (SmartFitter)")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}\DefaultIcon") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, str(ICON))
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\{PROGID}\shell\open\command") as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, command)
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\.mat\OpenWithProgids") as k:
        winreg.SetValueEx(k, PROGID, 0, winreg.REG_NONE, b"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--associate", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32":
        sys.exit("Shortcuts are Windows-only. Elsewhere run: python -m smartfitter")
    # The working directory must contain the smartfitter and nvfit packages.
    os.chdir(ROOT)
    if args.remove:
        for p in _shortcut_targets():
            p.unlink(missing_ok=True)
        _associate(remove=True)
        print("Removed SmartFitter shortcuts and the Open-with entry.")
        return
    for p in _shortcut_targets():
        _make_shortcut(p)
        print(f"Created {p}")
    if args.associate:
        _associate()
        print("Added SmartFitter to the Open-with list for .mat files.")


if __name__ == "__main__":
    main()
