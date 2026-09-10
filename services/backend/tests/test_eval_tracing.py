"""Tracing must be invisible when off and correct when on.

The judged sweep is ~8h and ~$3.60 of API spend; a span helper that raises
would take a run down with it, so the disabled and broken paths are tested
as carefully as the happy one.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import tracing  # noqa: E402


@pytest.fixture
def no_endpoint(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")


def test_disabled_without_endpoint(no_endpoint):
    assert tracing.enabled() is False


def test_span_yields_none_when_disabled(no_endpoint):
    with tracing.span("x", kind="CHAIN", **{"arm.name": "hyde"}) as s:
        assert s is None
    # The setters must tolerate that None rather than the call site guarding.
    tracing.set_input(None, "q")
    tracing.set_output(None, ["a"])
    tracing.set_documents(None, [{"chunk_id": 1, "text": "t"}])


def test_enabled_with_endpoint(endpoint):
    assert tracing.enabled() is True


def test_span_emits_attributes(recorder):
    with tracing.span("arm[hyde]", kind="CHAIN", **{"arm.name": "hyde"}) as s:
        tracing.set_input(s, "what is RAG?")
        tracing.set_output(s, ["a hypothetical passage"])
    spans = recorder.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert attrs["arm.name"] == "hyde"
    assert attrs["input.value"] == "what is RAG?"
    # Structured values become JSON so Phoenix renders them readably.
    assert attrs["output.value"] == '["a hypothetical passage"]'


def test_unknown_kind_falls_back_to_chain(recorder):
    with tracing.span("x", kind="NOT_A_KIND"):
        pass
    assert recorder.get_finished_spans()[0].attributes["openinference.span.kind"] == "CHAIN"


def test_none_attributes_are_dropped(recorder):
    with tracing.span("x", kind="CHAIN", **{"arm.name": None, "keep": "yes"}):
        pass
    attrs = recorder.get_finished_spans()[0].attributes
    assert "arm.name" not in attrs
    assert attrs["keep"] == "yes"


def test_documents_become_openinference_documents(recorder):
    with tracing.span("retrieve", kind="RETRIEVER") as s:
        tracing.set_documents(s, [
            {"chunk_id": 7, "doc_id": 3, "text": "hello", "score": 0.5},
            {"chunk_id": 9, "doc_id": 4, "text": "world"},
        ])
    attrs = recorder.get_finished_spans()[0].attributes
    assert attrs["retrieval.documents.0.document.id"] == "7"
    assert attrs["retrieval.documents.0.document.content"] == "hello"
    assert attrs["retrieval.documents.0.document.score"] == 0.5
    assert attrs["retrieval.documents.0.metadata.doc_id"] == "3"
    assert attrs["retrieval.documents.1.document.id"] == "9"
    # A chunk with no score must not invent one.
    assert "retrieval.documents.1.document.score" not in attrs


def test_set_documents_survives_a_bad_chunk(recorder):
    """A malformed chunk loses its span, not the run."""
    with tracing.span("retrieve", kind="RETRIEVER") as s:
        tracing.set_documents(s, [{"chunk_id": 1, "text": "ok", "score": "not-a-float"}])
    # Closed cleanly, no exception escaped.
    assert len(recorder.get_finished_spans()) == 1


@pytest.fixture
def recorder(endpoint, monkeypatch):
    """Point the module's tracer at an in-memory exporter.

    `get_finished_spans()` returns a snapshot rather than a live view, so the
    exporter itself is handed back and read *after* the span closes.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer(__name__))
    return exporter


# --- the span tree a judged run actually produces -------------------------
#
# The point of the issue was "I want to see query -> multiquery in Phoenix".
# These assert the shape that makes that visible, so a refactor that silently
# drops the transform span fails here rather than eight hours into a sweep.


class _FakeEngine:
    def _vector_candidates(self, query, top_k):
        return [{"chunk_id": hash(query) % 100, "doc_id": 1, "text": f"chunk for {query}"}]

    _bm25_candidates = _vector_candidates


class _FakeLLM:
    def invoke(self, messages):
        return type("R", (), {"content": "an answer"})()


def _arm_pipeline(arm, queries, mode="semantic"):
    from eval.pipeline import ArmPipeline

    return ArmPipeline(_FakeLLM(), _FakeEngine(), "sys", arm, queries, mode, top_k=5)


def test_multi_query_tree_shows_each_paraphrase(recorder, monkeypatch):
    monkeypatch.setenv("PAPERS_PLEASE_REPLAY", "")
    question = "what is RAG?"
    paraphrases = ["what is retrieval augmented generation?",
                   "how does RAG work?",
                   "define RAG"]
    pipeline = _arm_pipeline("multi_query", {question: paraphrases})
    pipeline.answer(question)

    names = [s.name for s in recorder.get_finished_spans()]
    # Transform first, then one span per paraphrase, then the fusion, all under
    # the arm root - which closes last.
    assert "arm.transform[multi_query]" in names
    assert [n for n in names if n.startswith("retrieve.sub[")] == [
        "retrieve.sub[0]", "retrieve.sub[1]", "retrieve.sub[2]"
    ]
    assert "retrieve.rrf_fuse" in names
    assert names[-1] == "arm[multi_query]"


def test_transform_span_carries_question_in_and_queries_out(recorder):
    question = "what is RAG?"
    paraphrases = ["a", "b", "c"]
    _arm_pipeline("multi_query", {question: paraphrases}).answer(question)

    transform = next(s for s in recorder.get_finished_spans()
                     if s.name == "arm.transform[multi_query]")
    assert transform.attributes["input.value"] == question
    assert transform.attributes["output.value"] == '["a", "b", "c"]'
    # The judged branch replays the free branch's cache; the span must say so.
    assert transform.attributes["arm.cached"] is True


def test_each_sub_query_span_shows_what_it_retrieved(recorder):
    question = "q"
    _arm_pipeline("multi_query", {question: ["alpha", "beta"]}).answer(question)

    subs = [s for s in recorder.get_finished_spans() if s.name.startswith("retrieve.sub[")]
    assert subs[0].attributes["input.value"] == "alpha"
    assert subs[0].attributes["retrieval.documents.0.document.content"] == "chunk for alpha"
    assert subs[1].attributes["input.value"] == "beta"
    assert subs[1].attributes["retrieval.documents.0.document.content"] == "chunk for beta"


def test_hyde_span_says_dense_even_when_mode_is_not(recorder):
    """HyDE ignores `mode`, so the span must not imply the run's mode applied."""
    question = "q"
    _arm_pipeline("hyde", {question: ["a hypothetical passage"]}, mode="bm25").answer(question)

    hyde = next(s for s in recorder.get_finished_spans() if s.name == "retrieve.hyde")
    assert hyde.attributes["retrieval.mode"] == "semantic"


def test_root_span_carries_the_sweep_coordinates(recorder):
    question = "q"
    _arm_pipeline("none", {question: [question]}).answer(question)

    root = next(s for s in recorder.get_finished_spans() if s.name == "arm[none]")
    assert root.attributes["arm.name"] == "none"
    assert root.attributes["retrieval.mode"] == "semantic"
    assert root.attributes["retrieval.top_k"] == 5
    assert root.attributes["input.value"] == question
    assert root.attributes["output.value"] == "an answer"


def test_single_query_arm_emits_one_retrieve_span(recorder):
    question = "q"
    _arm_pipeline("none", {question: [question]}).answer(question)

    names = [s.name for s in recorder.get_finished_spans()]
    assert "retrieve" in names
    assert not any(n.startswith("retrieve.sub[") for n in names)
    assert "retrieve.rrf_fuse" not in names


def test_pipeline_works_with_tracing_off(no_endpoint):
    """The sweep must not depend on Phoenix being up."""
    question = "q"
    result = _arm_pipeline("multi_query", {question: ["a", "b"]}).answer(question)
    assert result["answer"] == "an answer"
    assert result["contexts"]


def test_scalars_keep_their_type(recorder):
    """Stringified numbers can't be filtered numerically in Phoenix."""
    with tracing.span("x", kind="CHAIN",
                      **{"a.n": 5, "a.ratio": 0.5, "a.flag": True, "a.name": "s"}):
        pass
    attrs = recorder.get_finished_spans()[0].attributes
    assert attrs["a.n"] == 5 and isinstance(attrs["a.n"], int)
    assert attrs["a.ratio"] == 0.5
    # bool is a subclass of int; it must not arrive as 1.
    assert attrs["a.flag"] is True
    assert attrs["a.name"] == "s"
