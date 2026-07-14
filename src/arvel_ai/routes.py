"""HTTP routes this package can contribute — inert until wired in provider.boot().

Handlers use the same shapes as app route files: type the response with an
arvel Schema and it lands in the host app's OpenAPI at /schema.
"""

from __future__ import annotations

from typing import Any

from arvel import Route, Schema


class AiStatus(Schema):
    package: str
    status: str


async def status(request: Any) -> AiStatus:
    return AiStatus(package="arvel-ai", status="ok")


Route.get("/ai/status", status, name="ai.status")
