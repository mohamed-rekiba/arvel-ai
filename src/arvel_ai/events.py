"""Lifecycle events, dispatched through the host app's events dispatcher.

Apps hook auditing, budgets, or logging without arvel-ai building any of it:

    app.make("events").listen(AiResponseReceived, my_listener)
"""

from __future__ import annotations

from .contracts import ChatRequest, ChatResponse, EmbedRequest


class AiRequestSending:
    def __init__(self, driver: str, request: ChatRequest) -> None:
        self.driver = driver
        self.request = request


class AiResponseReceived:
    def __init__(self, driver: str, request: ChatRequest, response: ChatResponse) -> None:
        self.driver = driver
        self.request = request
        self.response = response


class AiEmbedding:
    def __init__(self, driver: str, request: EmbedRequest) -> None:
        self.driver = driver
        self.request = request
