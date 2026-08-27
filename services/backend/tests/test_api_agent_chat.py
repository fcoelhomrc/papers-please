"""Unit tests for POST /agent/chat.

Uses FastAPI's dependency override to swap get_agent() for a fixed instance
built from the fake chat model - same technique as SearchEngine's get_engine
would use, no ANTHROPIC_API_KEY or real model needed. Also covers the 503
path when the agent can't be built (e.g. missing API key), without leaving
get_agent overridden for that test.
"""
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import api
from orchestrator.graph import build_agent
from tests.fakes import FakeToolCallingModel


def test_chat_returns_reply_and_tool_calls_used():
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
    )
    resp2 = AIMessage(content="We already have papers on this, nothing fetched.")
    llm = FakeToolCallingModel(responses=[resp1, resp2])
    fake_agent = build_agent(llm)

    api.app.dependency_overrides[api.get_agent] = lambda: fake_agent
    try:
        with patch(
            "orchestrator.tools.get_status.func", return_value={"documents_total": 5}
        ):
            client = TestClient(api.app)
            response = client.post("/agent/chat", json={"message": "find papers on X"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "We already have papers on this, nothing fetched."
    assert body["tool_calls"] == ["get_status"]


def test_chat_503s_when_agent_cannot_be_built():
    api.app.dependency_overrides.pop(api.get_agent, None)
    api._agent = None
    with patch("api.build_agent", side_effect=RuntimeError("no ANTHROPIC_API_KEY")):
        client = TestClient(api.app)
        response = client.post("/agent/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "no ANTHROPIC_API_KEY" in response.json()["detail"]
