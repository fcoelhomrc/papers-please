"""Unit tests for eval/pipeline.py - no real LLM/search calls."""
import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.pipeline import AgenticPipeline, FixedPipeline
from orchestrator.graph import MAX_AGENT_RECURSION


class TestFixedPipeline:
    def test_answer_uses_reranked_search_results_as_context(self):
        search_response = MagicMock()
        search_response.results = [
            MagicMock(text="Chunk about muscle synergy."),
            MagicMock(text="Chunk about modular RL."),
        ]
        engine = MagicMock()
        engine.search.return_value = search_response

        llm = MagicMock()
        llm.invoke.return_value = AIMessage(content="It addresses X.")

        pipeline = FixedPipeline(
            llm, engine, system_prompt="Answer from the context only.", top_k=2, rerank=True
        )
        result = pipeline.answer("What does the muscle synergy paper address?")

        engine.search.assert_called_once_with(
            "What does the muscle synergy paper address?",
            top_k=2,
            rerank=True,
            rerank_top_k=2,
            # None unless a caller opts into the wide pool (#27); eval.run
            # passes config.search.rerank_candidates when it builds this.
            candidates=None,
        )
        assert result["answer"] == "It addresses X."
        assert result["contexts"] == ["Chunk about muscle synergy.", "Chunk about modular RL."]

        # the LLM call actually included the retrieved context, not just the question
        sent_prompt = llm.invoke.call_args.args[0][1]["content"]
        assert "Chunk about muscle synergy." in sent_prompt

        # ...and the configured system prompt, rather than one baked into the class
        assert llm.invoke.call_args.args[0][0]["content"] == "Answer from the context only."


class TestAgenticPipeline:
    def test_answer_extracts_contexts_from_search_chunks_tool_call(self):
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_chunks", "args": {"query": "x"}, "id": "call_1"}],
        )
        tool_result = ToolMessage(
            content=json.dumps([{"text": "Relevant passage one."}, {"text": "Relevant passage two."}]),
            tool_call_id="call_1",
        )
        final = AIMessage(content="Yes, per [doc 9, p3].")

        agent = MagicMock()
        agent.invoke.return_value = {
            "messages": [HumanMessage("has anyone studied X?"), tool_call_msg, tool_result, final]
        }

        pipeline = AgenticPipeline(agent)
        result = pipeline.answer("has anyone studied X?")

        assert result["answer"] == "Yes, per [doc 9, p3]."
        assert result["contexts"] == ["Relevant passage one.", "Relevant passage two."]

    def test_answer_with_no_search_calls_has_empty_contexts(self):
        final = AIMessage(content="Fetched 3 new papers.")
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [HumanMessage("fetch papers on X"), final]}

        pipeline = AgenticPipeline(agent)
        result = pipeline.answer("fetch papers on X")

        assert result["answer"] == "Fetched 3 new papers."
        assert result["contexts"] == []

    def test_answer_caps_recursion_limit(self):
        """Guardrail added after a real credit-exhaustion incident."""
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [AIMessage(content="done")]}

        AgenticPipeline(agent).answer("anything")

        _, kwargs = agent.invoke.call_args
        assert kwargs["config"]["recursion_limit"] == MAX_AGENT_RECURSION


def test_fixed_pipeline_forwards_its_candidate_pool():
    """#27 — the baseline retrieves as wide as the agent does, so a judged
    comparison stays about pipeline shape rather than about one of them
    accidentally having better retrieval settings."""
    from unittest.mock import MagicMock

    from eval.pipeline import FixedPipeline
    from schemas import SearchResponse

    engine = MagicMock()
    engine.search.return_value = SearchResponse(
        query="q", model="bge-small", mode="hybrid", reranked=True, results=[]
    )
    llm = MagicMock()
    llm.invoke.return_value.content = "an answer"

    FixedPipeline(llm, engine, system_prompt="sys", top_k=5, candidates=40).answer("q")

    assert engine.search.call_args.kwargs["candidates"] == 40
