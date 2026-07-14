"""A sample tool module the wiring test points config at."""

from __future__ import annotations

from arvel_ai.mcp import mcp_tool


@mcp_tool(description="Sample lookup used by the wiring test")
def sample_lookup(key: str) -> str:
    return f"value-for-{key}"
