#!/bin/bash
# SmartFitter for macOS. Double-click to install (first time only, ~1-2 min) and start SmartFitter.
# No Python install needed: uv (https://docs.astral.sh/uv/) downloads a private Python into
# ~/Library/Application Support/SmartFitter. Delete that folder to uninstall.
#
# Optional environment variables (for testing):
#   SMARTFITTER_REF=<branch|tag|commit>  install that GitHub ref instead of the latest release
#   SMARTFITTER_INSTALL_ONLY=1           install/update, check imports, then exit
set -euo pipefail

REPO="Divc09/Lab-Data-Analysis"
PYVER="3.14"
BASE="$HOME/Library/Application Support/SmartFitter"
APP="$BASE/app"
VENV="$BASE/venv"
export UV_INSTALL_DIR="$BASE/uv"
export UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR="$BASE/python"
export UV_CACHE_DIR="$BASE/cache"
mkdir -p "$BASE"

say() { printf '\n== %s\n' "$*"; }
fail() { printf '\nSmartFitter could not start: %s\nPress Return to close.' "$*"; read -r _ || true; exit 1; }
trap 'fail "an install step failed (see the messages above)."' ERR

# 1. uv (private copy, does not touch your shell settings)
find_uv() {
    UV="$(command -v uv || true)"
    for c in "$UV_INSTALL_DIR/uv" "$UV_INSTALL_DIR/bin/uv"; do
        if [ -z "$UV" ] && [ -x "$c" ]; then UV="$c"; fi
    done
}
find_uv
if [ -z "$UV" ]; then
    say "Downloading uv (one time)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    find_uv
    if [ -z "$UV" ]; then fail "uv did not install."; fi
fi

# 2. SmartFitter source: the latest GitHub release (or SMARTFITTER_REF)
REF="${SMARTFITTER_REF:-}"
if [ -z "$REF" ]; then
    REF="$(curl -fsSL --max-time 10 "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
        | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
fi
HAVE="$(cat "$APP/.ref" 2>/dev/null || true)"
if [ -z "$REF" ]; then
    if [ -z "$HAVE" ]; then fail "no internet connection, and SmartFitter is not installed yet."; fi
    echo "Offline: starting the installed version ($HAVE)."
elif [ "$REF" != "$HAVE" ]; then
    say "Downloading SmartFitter $REF"
    TMP="$(mktemp -d)"
    curl -fL --progress-bar "https://github.com/$REPO/archive/$REF.tar.gz" -o "$TMP/src.tar.gz"
    mkdir "$TMP/src" && tar -xzf "$TMP/src.tar.gz" -C "$TMP/src" --strip-components 1
    rm -rf "$APP" && mv "$TMP/src" "$APP" && rm -rf "$TMP"
    # The legacy Qt interface (PySide6) is not needed for SmartFitter v2.
    grep -vE '^(PySide6|QtAwesome)' "$APP/requirements.txt" > "$APP/requirements-mac.txt"
    say "Installing Python $PYVER and libraries (first time takes a minute or two)"
    if [ ! -x "$VENV/bin/python" ]; then "$UV" venv --python "$PYVER" "$VENV"; fi
    "$UV" pip install --python "$VENV/bin/python" -r "$APP/requirements-mac.txt"
    echo "$REF" > "$APP/.ref"
fi

cd "$APP"
if [ -n "${SMARTFITTER_INSTALL_ONLY:-}" ]; then
    "$VENV/bin/python" -c "import smartfitter.server, nvfit.fit_engine, webview; print('SmartFitter install OK')"
    exit 0
fi
trap - ERR
say "Starting SmartFitter $(cat .ref). Keep this window open (you can minimize it); closing it quits SmartFitter."
exec "$VENV/bin/python" -m smartfitter ${@+"$@"}
