"""Update notices (no download, no install). Network failures return {"status": "offline"}.

Git checkout : compare local HEAD with origin/main using `git ls-remote` (no fetch).
Frozen build : compare the latest GitHub release tag with core.APP_VERSION.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import core

REPO = "Divc09/Lab-Data-Analysis"
REMOTE_URL = f"https://github.com/{REPO}.git"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
RELEASE_API = f"https://api.github.com/repos/{REPO}/releases/latest"
UPDATE_COMMAND = "git pull --ff-only origin main"
TIMEOUT = 5
ROOT = Path(__file__).resolve().parents[1]
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def is_git_checkout() -> bool:
    return not is_frozen() and (ROOT / ".git").exists()


def _git(*args: str, timeout: float = TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          creationflags=_NOWIN)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text.split("-")[0])) or (0,)


def update_info() -> dict:
    """Static facts about this install (no network)."""
    mode = "frozen" if is_frozen() else "git" if is_git_checkout() else "source"
    info = {"mode": mode, "version": core.APP_VERSION, "repo": f"https://github.com/{REPO}",
            "releases_url": RELEASES_URL}
    if mode == "git":
        info["update_command"] = UPDATE_COMMAND
        info["path"] = str(ROOT)
        try:
            info["local_commit"] = _git("rev-parse", "HEAD", timeout=3).stdout.strip() or None
        except Exception:
            info["local_commit"] = None
    return info


def _check_git() -> dict:
    out = {"mode": "git", "update_command": UPDATE_COMMAND}
    local = _git("rev-parse", "HEAD", timeout=3).stdout.strip()
    r = _git("ls-remote", REMOTE_URL, "refs/heads/main")
    if r.returncode != 0 or not r.stdout.strip():
        return {"status": "offline", **out}
    remote = r.stdout.split()[0]
    out.update(local_commit=local, remote_commit=remote)
    if not local or local == remote:
        return {"status": "up_to_date", **out}
    # Remote commit already in our history (we are ahead): nothing to pull.
    if _git("cat-file", "-e", f"{remote}^{{commit}}", timeout=3).returncode == 0 and \
            _git("merge-base", "--is-ancestor", remote, local, timeout=3).returncode == 0:
        return {"status": "up_to_date", "ahead": True, **out}
    return {"status": "update_available", **out}


def _check_release() -> dict:
    out = {"mode": "frozen", "releases_url": RELEASES_URL}
    req = urllib.request.Request(RELEASE_API, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "SmartFitter-update-check"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return {"status": "no_release" if exc.code == 404 else "offline", **out}
    tag = str(data.get("tag_name", ""))
    out.update(latest=tag, url=data.get("html_url") or RELEASES_URL)
    newer = _version_tuple(tag) > _version_tuple(core.APP_VERSION)
    return {"status": "update_available" if newer else "up_to_date", **out}


def update_check() -> dict:
    """Never raises; any failure is reported as {'status': 'offline'}."""
    base = {"version": core.APP_VERSION}
    try:
        if is_frozen():
            return {**base, **_check_release()}
        if is_git_checkout():
            return {**base, **_check_git()}
        return {**base, "status": "unsupported", "mode": "source"}
    except Exception:
        return {**base, "status": "offline"}
