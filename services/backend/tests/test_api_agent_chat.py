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
from langgraph.checkpoint.memory import MemorySaver

import api
from orchestrator.graph import MAX_AGENT_RECURSION, build_agent
from tests.fakes import FakeToolCallingModel


def test_chat_returns_reply_and_tool_calls_used():
    resp1 = AIMessage(
        content="",
        tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
    )
    resp2 = AIMessage(content="We already have papers on this, nothing fetched.")
    llm = FakeToolCallingModel(responses=[resp1, resp2])
    fake_agent = build_agent(llm, checkpointer=MemorySaver())

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


def test_chat_remembers_prior_turns_in_same_thread():
    """Second call in the same thread_id sees the first turn's messages in
    its context - proves memory works, not just that a second call succeeds."""
    resp1 = AIMessage(content="Sure, I fetched papers on transformers.")
    resp2 = AIMessage(content="You just asked about transformers.")
    llm = FakeToolCallingModel(responses=[resp1, resp2])
    fake_agent = build_agent(llm, checkpointer=MemorySaver())

    api.app.dependency_overrides[api.get_agent] = lambda: fake_agent
    try:
        client = TestClient(api.app)
        r1 = client.post(
            "/agent/chat", json={"message": "fetch papers on transformers", "thread_id": "t1"}
        )
        r2 = client.post(
            "/agent/chat", json={"message": "what did I just ask?", "thread_id": "t1"}
        )
    finally:
        api.app.dependency_overrides.clear()

    assert r1.json()["reply"] == "Sure, I fetched papers on transformers."
    assert r2.json()["reply"] == "You just asked about transformers."
    # tool_calls for turn 2 must not include anything from turn 1
    assert r2.json()["tool_calls"] == []

    state = fake_agent.get_state({"configurable": {"thread_id": "t1"}})
    # 2 human + 2 AI = 4 messages retained across both turns
    assert len(state.values["messages"]) == 4


def test_chat_caps_recursion_so_a_runaway_loop_cant_burn_unbounded_credits():
    """Guardrail added after a real credit-exhaustion incident: every
    agent.invoke() call must pass recursion_limit, not rely on LangGraph's
    default (25 - way more than this agent legitimately needs)."""
    resp = AIMessage(content="done")
    llm = FakeToolCallingModel(responses=[resp])
    fake_agent = build_agent(llm, checkpointer=MemorySaver())

    api.app.dependency_overrides[api.get_agent] = lambda: fake_agent
    try:
        with patch.object(fake_agent, "invoke", wraps=fake_agent.invoke) as spy_invoke:
            client = TestClient(api.app)
            client.post("/agent/chat", json={"message": "hi"})
    finally:
        api.app.dependency_overrides.clear()

    _, kwargs = spy_invoke.call_args
    assert kwargs["config"]["recursion_limit"] == MAX_AGENT_RECURSION


def test_chat_503s_when_agent_cannot_be_built():
    """A missing API key must be a 503 with a readable reason, not a 500 or
    a traceback. Provider-neutral: which key is missing depends on
    llm.provider, and the agent is built lazily precisely so the rest of the
    app still starts without one."""
    api.app.dependency_overrides.pop(api.get_agent, None)
    api._agent = None
    with patch("api.make_agent_parts", side_effect=RuntimeError("no API key configured")):
        client = TestClient(api.app)
        response = client.post("/agent/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "no API key configured" in response.json()["detail"]


def test_chat_503s_when_the_provider_key_is_absent(monkeypatch):
    """The real shape of the above: no key in the environment at all."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PAPERS_PLEASE_REPLAY", raising=False)
    api.app.dependency_overrides.pop(api.get_agent, None)
    api._agent = None

    response = TestClient(api.app).post("/agent/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "OPENROUTER_API_KEY" in response.json()["detail"]
