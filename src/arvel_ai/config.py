"""Package config defaults, merged under the "ai" key.

The host app overrides any of these in its own config/ai.py — app values
always win. Read at runtime as config.get("ai.default") etc.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    # which driver `app.make("ai")` dispatches to by default
    "default": "memory",
    # per-driver settings live under drivers.<name>
    "drivers": {
        "memory": {},
        "fake": {},
    },
}
