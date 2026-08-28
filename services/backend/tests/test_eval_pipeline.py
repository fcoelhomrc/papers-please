"""Unit tests for eval/pipeline.py - no real LLM/search calls."""
import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval.pipeline import AgenticPipeline, FixedPipeline


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

        pipeline = FixedPipeline(llm, engine, top_k=2, rerank=True)
        result = pipeline.answer("What does the muscle synergy paper address?")

        engine.search.assert_called_once_with(
            "What does the muscle synergy paper address?", top_k=2, rerank=True, rerank_top_k=2
        )
        assert result["answer"] == "It addresses X."
        assert result["contexts"] == ["Chunk about muscle synergy.", "Chunk about modular RL."]

        # the LLM call actually included the retrieved context, not just the question
        sent_prompt = llm.invoke.call_args.args[0][1]["content"]
        assert "Chunk about muscle synergy." in sent_prompt


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
