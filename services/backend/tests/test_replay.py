"""Tests for offline replay (#30).

These drive the *real* LangGraph agent - the tool-calling loop, the message
sequence, the checkpointer - with both ends canned. That's the whole design:
if replay only faked the model, a scripted tool call would reach the real
tools and hit Pinecone; if it faked the whole agent, none of the graph wiring
these UI features depend on would be exercised at all.

No API key, no Postgres, no Pinecone, no network.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import Config, LLMConfig
from orchestrator.graph import MAX_AGENT_RECURSION, build_agent
from orchestrator.llm import make_agent_parts
from orchestrator.replay import (
    ReplayChatModel,
    ReplaySession,
    build_replay,
    load_fixtures,
    replay_enabled,
    select_fixture,
)

SEARCH_FIXTURE = {
    "name": "search-once",
    "match": "muscle synergy",
    "turns": [
        {"content": "", "tool_calls": [{"name": "search_chunks", "args": {"query": "muscle synergy"}}]},
        {"content": "Yes, per [doc 3, p4]."},
    ],
    "tool_results": {
        "search_chunks": [[{"doc_id": 3, "page_num": 4, "text": "A passage.", "score": 5.0}]]
    },
}

TWO_SEARCH_FIXTURE = {
    "name": "search-twice",
    "match": "attention",
    "turns": [
        {"content": "", "tool_calls": [{"name": "search_chunks", "args": {"query": "attention"}}]},
        {"content": "", "tool_calls": [{"name": "search_chunks", "args": {"query": "sparse attention"}}]},
        {"content": "Refined and answered."},
    ],
    "tool_results": {
        "search_chunks": [
            [{"doc_id": 1, "text": "too broad", "score": 1.0}],
            [{"doc_id": 2, "text": "the good one", "score": 8.0}],
        ]
    },
}

FALLBACK_FIXTURE = {
    "name": "_fallback",
    "match": "",
    "turns": [{"content": "No fixture for that."}],
}


def _agent(fixtures):
    llm, tools = build_replay(fixtures)
    return build_agent(llm, system_prompt="you are a test agent", tools=tools)


def _run(agent, question):
    return agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"recursion_limit": MAX_AGENT_RECURSION},
    )["messages"]


class TestReplayEnabled:
    def _cfg(self, provider="anthropic"):
        return Config(llm=LLMConfig(provider=provider))

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("PAPERS_PLEASE_REPLAY", raising=False)

        assert replay_enabled(self._cfg()) is False

    def test_config_provider_turns_it_on(self, monkeypatch):
        monkeypatch.delenv("PAPERS_PLEASE_REPLAY", raising=False)

        assert replay_enabled(self._cfg("replay")) is True

    def test_env_overrides_a_config_that_says_anthropic(self, monkeypatch):
        """The point of the env toggle: compose and the test suite must force
        replay without editing a checked-in config.yaml."""
        monkeypatch.setenv("PAPERS_PLEASE_REPLAY", "1")

        assert replay_enabled(self._cfg("anthropic")) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  OFF  "])
    def test_explicitly_disabled_values_mean_off(self, monkeypatch, value):
        """A variable set to "0" reads as disabled to anyone looking at it,
        so it must not switch replay on just by being non-empty."""
        monkeypatch.setenv("PAPERS_PLEASE_REPLAY", value)

        assert replay_enabled(self._cfg("anthropic")) is False


class TestSelectFixture:
    def test_matches_on_a_substring_of_the_question(self):
        got = select_fixture([SEARCH_FIXTURE, FALLBACK_FIXTURE], "what about muscle synergy?")

        assert got["name"] == "search-once"

    def test_case_insensitive(self):
        got = select_fixture([SEARCH_FIXTURE, FALLBACK_FIXTURE], "MUSCLE SYNERGY")

        assert got["name"] == "search-once"

    def test_longest_match_wins_regardless_of_file_order(self):
        """"fall recovery in legged robots" must not be answered by a fixture
        matching bare "robot"."""
        general = {"name": "general", "match": "robot", "turns": []}
        specific = {"name": "specific", "match": "fall recovery in legged robots", "turns": []}

        assert select_fixture([general, specific], "fall recovery in legged robots")["name"] == "specific"
        assert select_fixture([specific, general], "fall recovery in legged robots")["name"] == "specific"

    def test_unmatched_falls_back(self):
        got = select_fixture([SEARCH_FIXTURE, FALLBACK_FIXTURE], "something else entirely")

        assert got["name"] == "_fallback"

    def test_fallback_never_wins_on_its_empty_match_string(self):
        """`"" in anything` is True, so an unguarded loop would let the
        fallback claim every question before a real fixture got a look."""
        got = select_fixture([FALLBACK_FIXTURE, SEARCH_FIXTURE], "muscle synergy")

        assert got["name"] == "search-once"

    def test_no_fixtures_at_all_is_none_not_a_crash(self):
        assert select_fixture([], "anything") is None


class TestReplayAgentEndToEnd:
    def test_scripted_tool_call_reaches_the_replay_tool_and_comes_back(self):
        messages = _run(_agent([SEARCH_FIXTURE, FALLBACK_FIXTURE]), "muscle synergy?")

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0].content)[0]["doc_id"] == 3
        assert messages[-1].content == "Yes, per [doc 3, p4]."

    def test_successive_calls_walk_the_result_list(self):
        """The second search must return the second canned result - a fixture
        that refines its query is meaningless if both calls answer the same."""
        messages = _run(_agent([TWO_SEARCH_FIXTURE, FALLBACK_FIXTURE]), "attention?")

        texts = [
            json.loads(m.content)[0]["text"]
            for m in messages
            if isinstance(m, ToolMessage)
        ]
        assert texts == ["too broad", "the good one"]

    def test_a_new_question_restarts_the_tool_counters(self):
        """Without a reset, a second conversation in the same process reads
        its fixture's tool results from wherever the first left off."""
        agent = _agent([TWO_SEARCH_FIXTURE, FALLBACK_FIXTURE])

        _run(agent, "attention?")
        messages = _run(agent, "attention again?")

        first_result = next(m for m in messages if isinstance(m, ToolMessage))
        assert json.loads(first_result.content)[0]["text"] == "too broad"

    def test_unmatched_question_still_answers(self):
        messages = _run(_agent([SEARCH_FIXTURE, FALLBACK_FIXTURE]), "unrelated question")

        assert messages[-1].content == "No fixture for that."

    def test_running_past_the_script_ends_the_turn_instead_of_raising(self):
        """A fixture that calls a tool but never scripts a final answer would
        otherwise loop to the recursion limit. Worse, raising mid-loop
        checkpoints a tool_use with no tool_result, which permanently breaks
        the thread - the exact incident orchestrator/tools.py documents."""
        truncated = {
            "name": "truncated",
            "match": "truncated",
            "turns": [
                {"content": "", "tool_calls": [{"name": "get_status", "args": {}}]}
            ],
            "tool_results": {"get_status": [{"documents_total": 1}]},
        }

        messages = _run(_agent([truncated]), "truncated case")

        assert isinstance(messages[-1], AIMessage)
        assert "exhausted" in messages[-1].content

    def test_a_missing_tool_result_returns_an_error_not_an_exception(self):
        broken = {
            "name": "broken",
            "match": "broken",
            "turns": [
                {"content": "", "tool_calls": [{"name": "get_document", "args": {"doc_id": 1}}]},
                {"content": "handled"},
            ],
            "tool_results": {},
        }

        messages = _run(_agent([broken]), "broken case")

        tool_message = next(m for m in messages if isinstance(m, ToolMessage))
        assert "no replay result" in tool_message.content
        assert messages[-1].content == "handled"


class TestReplaySession:
    def test_extra_calls_repeat_the_last_result_rather_than_erroring(self):
        session = ReplaySession([SEARCH_FIXTURE])
        session.observe("muscle synergy")

        first = session.next_result("search_chunks")
        second = session.next_result("search_chunks")

        assert first == second

    def test_observing_the_same_question_does_not_reset_counters(self):
        """The model calls observe() on every turn, so a reset there would
        rewind the tool results in the middle of a conversation."""
        session = ReplaySession([TWO_SEARCH_FIXTURE])
        session.observe("attention")
        session.next_result("search_chunks")
        session.observe("attention")

        assert session.next_result("search_chunks")[0]["text"] == "the good one"


class TestMakeAgentParts:
    def test_replay_swaps_both_the_model_and_the_tools(self, monkeypatch):
        """Swapping only the model would send scripted tool calls straight at
        the real Pinecone-backed tools."""
        monkeypatch.setenv("PAPERS_PLEASE_REPLAY", "1")

        llm, tools = make_agent_parts(Config())

        assert isinstance(llm, ReplayChatModel)
        assert {t.name for t in tools} == {
            "fetch_papers",
            "get_status",
            "search_chunks",
            "get_document",
        }

    def test_normal_mode_keeps_the_real_tools(self, monkeypatch):
        monkeypatch.delenv("PAPERS_PLEASE_REPLAY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

        _, tools = make_agent_parts(Config())

        assert tools is None


class TestShippedFixtures:
    """The fixtures under eval/replay/ are what someone running the app with
    no API key actually sees, so they get checked like code."""

    def test_all_parse_and_carry_the_required_shape(self):
        fixtures = load_fixtures()

        assert fixtures, "no fixtures found in eval/replay/"
        for f in fixtures:
            assert f.get("name"), f
            assert f.get("turns"), f["name"]
            assert isinstance(f.get("description", ""), str)

    def test_exactly_one_fallback(self):
        names = [f["name"] for f in load_fixtures()]

        assert names.count("_fallback") == 1

    def test_every_scripted_tool_call_has_a_result_to_return(self):
        """A tool call with no canned result answers with an error string.
        That is correct behaviour for a *missing* fixture and a bug in a
        shipped one."""
        for f in load_fixtures():
            called = {
                c["name"] for turn in f["turns"] for c in (turn.get("tool_calls") or [])
            }
            available = set((f.get("tool_results") or {}).keys())
            assert called <= available, f"{f['name']}: {called - available} unanswered"

    def test_each_fixture_ends_with_a_spoken_answer(self):
        """A fixture whose last turn is a tool call runs off the end of its
        script and shows the user the exhausted-script message."""
        for f in load_fixtures():
            assert f["turns"][-1].get("content"), f"{f['name']} ends on a tool call"

    def test_every_fixture_actually_reachable_by_its_own_match(self):
        fixtures = load_fixtures()
        for f in fixtures:
            if f["name"] == "_fallback":
                continue
            chosen = select_fixture(fixtures, f["match"])
            assert chosen["name"] == f["name"], (
                f"{f['name']} is shadowed by {chosen['name']}"
            )
