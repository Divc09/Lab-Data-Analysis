"""Endpoints: update_info (local facts) and update_check (network, 5 s timeout, silent on failure)."""
from __future__ import annotations

from . import updates

ENDPOINTS = {
    "update_info": lambda ws: updates.update_info(),
    "update_check": lambda ws: updates.update_check(),
}
