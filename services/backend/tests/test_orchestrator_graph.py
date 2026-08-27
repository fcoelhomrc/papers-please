"""Unit tests for orchestrator/graph.py and llm.py.

No API key needed: llm.py is tested by checking make_llm() picks the right
class for a given cfg.llm.provider (constructor mocked, no real client/key
required). graph.py is tested by driving build_agent() with the fake model
from tests/fakes.py, so it's the real create_agent loop + real tool
functions (with the DB/network layer inside get_status mocked) - only the
LLM's "reasoning" is scripted, since that's the one thing we can't test
without an actual model.
"""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.graph import build_agent
from orchestrator.llm import make_llm
from tests.fakes import FakeToolCallingModel


class TestMakeLLM:
    def test_anthropic_provider(self):
        cfg = MagicMock()
        cfg.llm.provider = "anthropic"
        cfg.llm.model = "claude-haiku-4-5"
        with patch("orchestrator.llm.ChatAnthropic") as MockAnthropic:
            make_llm(cfg)
            MockAnthropic.assert_called_once_with(model="claude-haiku-4-5")

    def test_vllm_provider(self):
        cfg = MagicMock()
        cfg.llm.provider = "vllm"
        cfg.llm.vllm_url = "http://localhost:8001/v1"
        cfg.llm.vllm_model = "my-local-model"
        with patch("orchestrator.llm.ChatOpenAI") as MockOpenAI:
            make_llm(cfg)
            MockOpenAI.assert_called_once_with(
                base_url="http://localhost:8001/v1",
                api_key="EMPTY",
                model="my-local-model",
            )

    def test_unknown_provider_raises(self):
        cfg = MagicMock()
        cfg.llm.provider = "made-up"
        try:
            make_llm(cfg)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestBuildAgent:
    def test_calls_get_status_then_stops_when_nothing_pending(self):
        """Scripts the LLM: turn 1 calls get_status, turn 2 gives a final answer."""
        resp1 = AIMessage(
            content="",
            tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
        )
        resp2 = AIMessage(content="Nothing pending, all done.")
        llm = FakeToolCallingModel(responses=[resp1, resp2])

        with patch(
            "orchestrator.tools.get_status.func",
            return_value={"documents_total": 0, "chunks_pending_embed": 0},
        ):
            agent = build_agent(llm)
            result = agent.invoke({"messages": [HumanMessage("check the pipeline")]})

        messages = result["messages"]
        # Human -> AI(tool_call) -> Tool(result) -> AI(final)
        assert isinstance(messages[0], HumanMessage)
        assert messages[1].tool_calls[0]["name"] == "get_status"
        assert messages[2].content == '{"documents_total": 0, "chunks_pending_embed": 0}'
        assert messages[-1].content == "Nothing pending, all done."

    def test_routes_to_embed_when_status_says_chunks_are_pending(self):
        """Scripts the LLM to inspect get_status's output and pick embed_pending -
        this is the actual routing behaviour get_status exists to support."""
        resp1 = AIMessage(
            content="",
            tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
        )
        resp2 = AIMessage(
            content="",
            tool_calls=[{"name": "embed_pending", "args": {"limit": 50}, "id": "call_2"}],
        )
        resp3 = AIMessage(content="Embedded the pending chunks.")
        llm = FakeToolCallingModel(responses=[resp1, resp2, resp3])

        with (
            patch(
                "orchestrator.tools.get_status.func",
                return_value={"chunks_pending_embed": 12},
            ),
            patch("orchestrator.tools.PdfEmbedder") as MockEmbedder,
        ):
            agent = build_agent(llm)
            result = agent.invoke({"messages": [HumanMessage("check the pipeline")]})

            MockEmbedder.return_value.execute.assert_called_once_with(max_chunks=50)

        assert result["messages"][-1].content == "Embedded the pending chunks."
