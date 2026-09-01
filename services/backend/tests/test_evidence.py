"""Tests for orchestrator/evidence.py (#31) — turning an agent run's message
list into citation cards and a tool trace.

Pure functions over LangChain message objects. No API, no DB.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orchestrator.evidence import extract_contexts, extract_evidence, extract_trace


def _call(name, call_id, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _result(call_id, payload, name="search_chunks"):
    return ToolMessage(content=json.dumps(payload), tool_call_id=call_id, name=name)


def _chunk(chunk_id, doc_id=1, score=1.0, text="passage", title="A Paper", page=1):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title,
        "authors": ["A. Author"],
        "year": 2024,
        "page_num": page,
        "text": text,
        "score": score,
    }


class TestExtractEvidence:
    def test_pulls_the_chunks_a_search_returned(self):
        messages = [
            HumanMessage("q"),
            _call("search_chunks", "c1", query="q"),
            _result("c1", [_chunk(10, doc_id=3, page=4)]),
            AIMessage(content="Yes, per [doc 3, p4]."),
        ]

        [got] = extract_evidence(messages)

        assert got["chunk_id"] == 10
        assert got["doc_id"] == 3
        assert got["page_num"] == 4
        assert got["title"] == "A Paper"

    def test_omits_chunk_text(self):
        """It's already in the model's answer; shipping it again multiplies
        the response size for a card that shows a title and a page."""
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [_chunk(10, text="a very long passage " * 50)]),
        ]

        assert "text" not in extract_evidence(messages)[0]

    def test_preserves_the_order_the_agent_saw(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [_chunk(30), _chunk(10), _chunk(20)]),
        ]

        assert [e["chunk_id"] for e in extract_evidence(messages)] == [30, 10, 20]

    def test_a_chunk_found_by_two_searches_is_one_citation(self):
        """A refined second query usually re-surfaces the same strong chunk.
        Listing it twice would present one piece of evidence as corroboration
        by two."""
        messages = [
            _call("search_chunks", "c1", query="broad"),
            _result("c1", [_chunk(10, score=2.0)]),
            _call("search_chunks", "c2", query="refined"),
            _result("c2", [_chunk(10, score=8.0), _chunk(11)]),
        ]

        got = extract_evidence(messages)

        assert [e["chunk_id"] for e in got] == [10, 11]

    def test_deduplication_keeps_the_better_score(self):
        messages = [
            _call("search_chunks", "c1", query="broad"),
            _result("c1", [_chunk(10, score=2.0)]),
            _call("search_chunks", "c2", query="refined"),
            _result("c2", [_chunk(10, score=8.0)]),
        ]

        assert extract_evidence(messages)[0]["score"] == 8.0

    def test_ignores_results_from_other_tools(self):
        """get_document returns a paper's metadata, not a retrieved passage -
        citing it would claim the agent found evidence it never searched
        for."""
        messages = [
            _call("get_document", "c1", doc_id=3),
            ToolMessage(
                content=json.dumps({"doc_id": 3, "title": "A Paper", "abstract": "..."}),
                tool_call_id="c1",
                name="get_document",
            ),
        ]

        assert extract_evidence(messages) == []

    def test_a_failed_search_contributes_nothing(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [{"error": "search failed (connection pool exhausted)"}]),
        ]

        assert extract_evidence(messages) == []

    def test_an_empty_search_contributes_nothing(self):
        messages = [_call("search_chunks", "c1", query="q"), _result("c1", [])]

        assert extract_evidence(messages) == []

    def test_non_json_tool_content_is_not_fatal(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            ToolMessage(content="not json at all", tool_call_id="c1", name="search_chunks"),
        ]

        assert extract_evidence(messages) == []

    def test_a_run_with_no_tool_calls_has_no_evidence(self):
        assert extract_evidence([HumanMessage("hi"), AIMessage(content="hello")]) == []


class TestExtractTrace:
    def test_records_the_tool_its_arguments_and_what_came_back(self):
        messages = [
            _call("search_chunks", "c1", query="fall recovery", top_k=5),
            _result("c1", [_chunk(1), _chunk(2)]),
        ]

        [step] = extract_trace(messages)

        assert step["tool"] == "search_chunks"
        assert step["args"] == {"query": "fall recovery", "top_k": 5}
        assert step["summary"] == "2 chunks"
        assert step["ok"] is True

    def test_singular_summary_for_one_chunk(self):
        messages = [_call("search_chunks", "c1", query="q"), _result("c1", [_chunk(1)])]

        assert extract_trace(messages)[0]["summary"] == "1 chunk"

    def test_a_failure_is_marked_not_ok_and_says_why(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [{"error": "search failed (pool exhausted)"}]),
        ]

        [step] = extract_trace(messages)

        assert step["ok"] is False
        assert "pool exhausted" in step["summary"]

    def test_get_document_summarises_as_the_paper_title(self):
        messages = [
            _call("get_document", "c1", doc_id=3),
            ToolMessage(
                content=json.dumps({"doc_id": 3, "title": "Learning Agile Recovery"}),
                tool_call_id="c1",
                name="get_document",
            ),
        ]

        assert extract_trace(messages)[0]["summary"] == "Learning Agile Recovery"

    def test_an_unanswered_call_is_still_listed(self):
        """An unanswered tool call means the run was cut short - a recursion
        limit or a crash - which is exactly when someone reads a trace."""
        messages = [_call("search_chunks", "c1", query="q")]

        [step] = extract_trace(messages)

        assert step["ok"] is False
        assert "no result" in step["summary"]

    def test_lists_every_call_in_order(self):
        messages = [
            _call("get_status", "c1"),
            ToolMessage(content=json.dumps({"documents_total": 5}), tool_call_id="c1", name="get_status"),
            _call("search_chunks", "c2", query="q"),
            _result("c2", [_chunk(1)]),
        ]

        assert [s["tool"] for s in extract_trace(messages)] == ["get_status", "search_chunks"]

    def test_a_run_with_no_tool_calls_has_an_empty_trace(self):
        assert extract_trace([HumanMessage("hi"), AIMessage(content="hello")]) == []


class TestExtractContexts:
    def test_returns_texts_and_doc_ids_for_scoring(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [_chunk(1, doc_id=3, text="first"), _chunk(2, doc_id=7, text="second")]),
        ]

        contexts, doc_ids = extract_contexts(messages)

        assert contexts == ["first", "second"]
        assert doc_ids == [3, 7]

    def test_does_not_deduplicate(self):
        """Unlike the citation cards: eval measures what the pipeline actually
        put in front of the model, and collapsing a repeat would understate
        the context the answer came from."""
        messages = [
            _call("search_chunks", "c1", query="a"),
            _result("c1", [_chunk(1, text="same")]),
            _call("search_chunks", "c2", query="b"),
            _result("c2", [_chunk(1, text="same")]),
        ]

        contexts, _ = extract_contexts(messages)

        assert contexts == ["same", "same"]

    def test_survives_a_failed_search(self):
        messages = [
            _call("search_chunks", "c1", query="q"),
            _result("c1", [{"error": "boom"}]),
        ]

        assert extract_contexts(messages) == ([], [])
