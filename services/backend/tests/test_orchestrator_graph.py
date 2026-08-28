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
        cfg.llm.max_tokens = 512
        with patch("orchestrator.llm.ChatAnthropic") as MockAnthropic:
            make_llm(cfg)
            MockAnthropic.assert_called_once_with(model="claude-haiku-4-5", max_tokens=512)

    def test_vllm_provider(self):
        cfg = MagicMock()
        cfg.llm.provider = "vllm"
        cfg.llm.vllm_url = "http://localhost:8001/v1"
        cfg.llm.vllm_model = "my-local-model"
        cfg.llm.max_tokens = 512
        with patch("orchestrator.llm.ChatOpenAI") as MockOpenAI:
            make_llm(cfg)
            MockOpenAI.assert_called_once_with(
                base_url="http://localhost:8001/v1",
                api_key="EMPTY",
                model="my-local-model",
                max_tokens=512,
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
    def test_calls_get_status_then_stops_when_nothing_to_fetch(self):
        """Scripts the LLM: check status, decide we already have enough, stop."""
        resp1 = AIMessage(
            content="",
            tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
        )
        resp2 = AIMessage(content="We already have papers on this topic, nothing to fetch.")
        llm = FakeToolCallingModel(responses=[resp1, resp2])

        with patch(
            "orchestrator.tools.get_status.func",
            return_value={"documents_total": 40},
        ):
            agent = build_agent(llm)
            result = agent.invoke({"messages": [HumanMessage("find papers on transformers")]})

        messages = result["messages"]
        # Human -> AI(tool_call) -> Tool(result) -> AI(final)
        assert isinstance(messages[0], HumanMessage)
        assert messages[1].tool_calls[0]["name"] == "get_status"
        assert messages[2].content == '{"documents_total": 40}'
        assert messages[-1].content == "We already have papers on this topic, nothing to fetch."

    def test_calls_fetch_papers_with_llm_chosen_query(self):
        """Scripts the LLM to check status, then decide to fetch - the one
        genuinely agentic decision this agent makes (what to search for)."""
        resp1 = AIMessage(
            content="",
            tool_calls=[{"name": "get_status", "args": {}, "id": "call_1"}],
        )
        resp2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_papers",
                    "args": {"query": "transformer attention mechanisms", "max_papers": 30},
                    "id": "call_2",
                }
            ],
        )
        resp3 = AIMessage(content="Fetched new papers on transformers.")
        llm = FakeToolCallingModel(responses=[resp1, resp2, resp3])

        with (
            patch("orchestrator.tools.get_status.func", return_value={"documents_total": 0}),
            patch("orchestrator.tools.SemanticScholarFetcher") as MockFetcher,
        ):
            MockFetcher.return_value.fetch.return_value = 30
            agent = build_agent(llm)
            result = agent.invoke({"messages": [HumanMessage("find papers on transformers")]})

            MockFetcher.return_value.fetch.assert_called_once_with(
                query="transformer attention mechanisms", max_papers=30
            )

        assert result["messages"][-1].content == "Fetched new papers on transformers."

    def test_answers_from_search_chunks_instead_of_fetching(self):
        """Scripts the LLM to search existing papers and answer from them -
        the retrieval half of the agent, not the fetch half."""
        resp1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_chunks",
                    "args": {"query": "transformers for cervical cancer survival", "top_k": 5},
                    "id": "call_1",
                }
            ],
        )
        resp2 = AIMessage(
            content='Yes - doc_id 9, page 3 used a transformer to predict survival.'
        )
        llm = FakeToolCallingModel(responses=[resp1, resp2])

        search_result = [
            {
                "doc_id": 9,
                "title": "Deep Learning for Cervical Cancer Survival",
                "authors": ["A. Author"],
                "year": 2022,
                "page_num": 3,
                "text": "We used a transformer model to predict survival...",
                "score": 0.87,
            }
        ]

        with patch("orchestrator.tools.search_chunks.func", return_value=search_result):
            agent = build_agent(llm)
            result = agent.invoke({
                "messages": [HumanMessage(
                    "has anyone used transformers to predict cervical cancer survival?"
                )]
            })

        messages = result["messages"]
        assert messages[1].tool_calls[0]["name"] == "search_chunks"
        assert "doc_id" in messages[-1].content and "9" in messages[-1].content
