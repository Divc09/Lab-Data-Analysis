"""PyInstaller entry point for the portable SmartFitter build (same CLI as `python -m smartfitter`)."""
import multiprocessing

from smartfitter.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
