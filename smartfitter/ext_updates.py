"""Endpoints: update_info (local facts) and update_check (network, 5 s timeout, silent on failure)."""

from __future__ import annotations

from . import updates, core

ENDPOINTS = {
    "update_info": core.plain_endpoint(lambda ws: updates.update_info()),
    "update_check": core.plain_endpoint(lambda ws: updates.update_check()),
}
