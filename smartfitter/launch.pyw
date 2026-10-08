"""Windowless entry point used by the shortcuts and the .mat Open-with entry.

It works from any working directory (e.g. when Windows opens a .mat file).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartfitter.__main__ import main  # noqa: E402

main()
