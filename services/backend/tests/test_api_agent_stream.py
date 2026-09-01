"""End-to-end API tests for the evidence/trace response and the SSE stream
(#31), driven through offline replay (#30).

TestClient(app) skips the lifespan, so no OTel connection and no search
engine pre-warm; PAPERS_PLEASE_REPLAY makes the agent itself fixture-backed.
Nothing here needs an API key, Postgres or Pinecone.
"""
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PAPERS_PLEASE_REPLAY", "1")
    import api

    # The agent is cached in a module global, so a test that ran before this
    # fixture set the env var would otherwise leave a real ChatAnthropic
    # behind for everything after it.
    monkeypatch.setattr(api, "_agent", None)
    return TestClient(api.app)


def _sse_events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs."""
    events, name = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name:
            events.append((name, json.loads(line[len("data: ") :])))
            name = None
    return events


class TestChatEvidence:
    def test_answer_carries_the_chunks_behind_its_citations(self, client):
        r = client.post(
            "/agent/chat",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "a"},
        )

        body = r.json()
        assert r.status_code == 200
        assert [e["doc_id"] for e in body["evidence"]] == [3, 7]
        assert body["evidence"][0]["page_num"] == 4
        # the prose cites [doc 3, p4]; the card is what makes that openable
        assert "[doc 3, p4]" in body["reply"]

    def test_trace_says_what_was_searched_for(self, client):
        r = client.post(
            "/agent/chat",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "b"},
        )

        [step] = r.json()["trace"]
        assert step["tool"] == "search_chunks"
        assert "fall recovery" in step["args"]["query"]
        assert step["ok"] is True

    def test_a_refined_second_search_shows_as_two_steps(self, client):
        r = client.post(
            "/agent/chat", json={"message": "what about transformer attention?", "thread_id": "c"}
        )

        trace = r.json()["trace"]
        assert len(trace) == 2
        assert trace[0]["args"]["query"] != trace[1]["args"]["query"]

    def test_abstention_returns_no_evidence(self, client):
        """Retrieval found nothing, so there is nothing to cite - and the UI
        needs to be able to tell this apart from a failure."""
        r = client.post(
            "/agent/chat", json={"message": "anything on protein folding?", "thread_id": "d"}
        )

        body = r.json()
        assert body["evidence"] == []
        assert body["trace"][0]["ok"] is True  # the search worked; it just found nothing
        assert body["trace"][0]["summary"] == "0 chunks"

    def test_a_tool_failure_is_visible_in_the_trace(self, client):
        r = client.post(
            "/agent/chat", json={"message": "tell me about muscle synergy", "thread_id": "e"}
        )

        [step] = r.json()["trace"]
        assert step["ok"] is False
        assert "connection pool exhausted" in step["summary"]

    def test_evidence_covers_only_the_current_turn(self, client):
        """With memory, the agent's state holds the whole conversation.
        Citing three turns' worth of evidence under one answer would
        attribute sources to claims that never used them."""
        client.post(
            "/agent/chat",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "f"},
        )
        r = client.post(
            "/agent/chat", json={"message": "anything on protein folding?", "thread_id": "f"}
        )

        assert r.json()["evidence"] == []

    def test_tool_calls_still_present_for_existing_callers(self, client):
        r = client.post(
            "/agent/chat",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "g"},
        )

        assert r.json()["tool_calls"] == ["search_chunks"]


class TestChatStream:
    def test_emits_a_step_per_tool_call_then_a_done(self, client):
        r = client.post(
            "/agent/chat/stream",
            json={"message": "what about transformer attention?", "thread_id": "h"},
        )

        events = _sse_events(r)
        assert r.status_code == 200
        assert events[-1][0] == "done"
        tool_calls = [d for name, d in events if name == "step" and d["kind"] == "tool_call"]
        assert [d["tool"] for d in tool_calls] == ["search_chunks", "search_chunks"]

    def test_tool_arguments_arrive_with_the_step(self, client):
        """The point of streaming here: a search announces what it is looking
        for while it runs, rather than after it no longer matters."""
        r = client.post(
            "/agent/chat/stream",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "i"},
        )

        first = next(
            d for name, d in _sse_events(r) if name == "step" and d["kind"] == "tool_call"
        )
        assert "fall recovery" in first["args"]["query"]

    def test_done_carries_the_same_payload_as_the_plain_endpoint(self, client):
        plain = client.post(
            "/agent/chat",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "j"},
        ).json()
        streamed = client.post(
            "/agent/chat/stream",
            json={"message": "has anyone studied fall recovery in legged robots?", "thread_id": "k"},
        )

        done = next(d for name, d in _sse_events(streamed) if name == "done")
        assert done["reply"] == plain["reply"]
        assert done["evidence"] == plain["evidence"]
        assert done["trace"] == plain["trace"]

    def test_declares_itself_as_an_event_stream_and_disables_proxy_buffering(self, client):
        """nginx proxies /api and buffers by default, which would deliver the
        whole stream as one lump - indistinguishable from not streaming."""
        r = client.post(
            "/agent/chat/stream", json={"message": "anything on protein folding?", "thread_id": "l"}
        )

        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["x-accel-buffering"] == "no"

    def test_a_mid_stream_failure_is_reported_in_band(self, client, monkeypatch):
        """The response has already begun, so a status code is no longer
        available - without an in-band error the client waits forever on a
        `done` that never comes."""
        import api

        agent = api.get_agent()

        async def boom(*a, **kw):
            raise RuntimeError("graph exploded")
            yield  # pragma: no cover - unreachable; makes this an async generator

        monkeypatch.setattr(agent, "astream", boom)

        r = client.post(
            "/agent/chat/stream", json={"message": "anything at all", "thread_id": "m"}
        )

        events = _sse_events(r)
        assert events[-1][0] == "error"
        assert "graph exploded" in events[-1][1]["detail"]
