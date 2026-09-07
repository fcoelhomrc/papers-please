"""Unit tests for eval/run.py's harness wiring - ragas.evaluate() itself is
mocked (a real call costs judge tokens per metric per question, not something
to do on every test run).

Marked `eval` and excluded from default collection entirely (not just
marker-deselected - see pyproject.toml's --ignore): importing ragas triggers
nest_asyncio.apply() at import time (ragas/executor.py, unconditional), which
poisons the process-wide asyncio event loop and breaks FastAPI TestClient
(anyio-based) tests if both run in the same pytest process - and pytest
imports every collected file to read its markers, so marker-based deselection
alone isn't enough. Run in isolation with:

    uv run pytest -o addopts="" tests/test_eval_run.py -m eval
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eval.run import answer_all, is_abstention, run_eval, score_retrieval, stratified_sample

pytestmark = pytest.mark.eval


def row(qid, question="q", chunks=(1,), synth="single_hop_specifc_query_synthesizer"):
    return {
        "id": qid,
        "question": question,
        "reference": f"gold for {qid}",
        "reference_chunk_ids": list(chunks),
        "synthesizer": synth,
    }


class TestAnswerAll:
    def test_shapes_pipeline_output_into_ragas_records(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = {
            "answer": "the answer", "contexts": ["c1", "c2"], "chunk_ids": [7, 8]
        }

        records, retrieved = answer_all(pipeline, [row("q0001", "why?")])

        assert records == [
            {
                "user_input": "why?",
                "response": "the answer",
                "retrieved_contexts": ["c1", "c2"],
                "reference": "gold for q0001",
            }
        ] and retrieved == [[7, 8]]

    def test_substitutes_a_placeholder_when_nothing_was_retrieved(self):
        """ragas requires non-empty retrieved_contexts even when retrieval
        genuinely found nothing; an explicit placeholder beats crashing."""
        pipeline = MagicMock()
        pipeline.answer.return_value = {"answer": "nothing found", "contexts": [], "chunk_ids": []}

        records, _ = answer_all(pipeline, [row("q0001")])

        assert records[0]["retrieved_contexts"] == ["(no context retrieved)"]

    def test_survives_one_question_failing(self):
        """Regression test for a real incident: one question hit LangGraph's
        recursion limit and crashed the whole loop before anything reached
        disk, discarding ~40 other questions' already-paid-for answers."""
        pipeline = MagicMock()
        pipeline.answer.side_effect = [
            {"answer": "a1", "contexts": ["c1"], "chunk_ids": [1]},
            RuntimeError("recursion limit hit"),
            {"answer": "a3", "contexts": ["c3"], "chunk_ids": [3]},
        ]

        records, retrieved = answer_all(
            pipeline, [row("q1"), row("q2"), row("q3")]
        )

        assert len(records) == 3
        assert "recursion limit hit" in records[1]["response"]
        assert retrieved == [[1], [], [3]]


class TestAbstention:
    """ResponseRelevancy scores a noncommittal answer 0, so a correct
    abstention is punished by it. These rows are separated rather than
    silently depressing the mean."""

    def test_recognises_a_refusal(self):
        assert is_abstention("The context does not contain that information.")

    def test_a_real_answer_is_not_an_abstention(self):
        assert not is_abstention("RAGCache reaches 4x lower latency than vLLM.")

    def test_tolerates_an_empty_answer(self):
        assert not is_abstention("")


class TestScoreRetrieval:
    def test_scores_against_chunk_ids_not_documents(self):
        """The curated set labels chunks. Document-level scoring cannot say
        whether the right passage was found."""
        out = score_retrieval([row("q1", chunks=(5,))], [[5, 9]])

        assert out["recall"] == 1.0

    def test_skips_questions_with_no_labels(self):
        assert score_retrieval([row("q1", chunks=())], [[1]]) == {}

    def test_reports_an_interval_on_every_metric(self):
        out = score_retrieval([row("q1", chunks=(1,)), row("q2", chunks=(2,))], [[1], [9]])

        assert "recall_ci" in out and out["n"] == 2


class TestStratifiedSample:
    def test_spreads_across_synthesizers(self):
        """46/33/21 across question types that score very differently - an
        unstratified sample of 10 reports whichever type it drew."""
        rows = [row(f"s{i}", synth="single_hop_specifc_query_synthesizer") for i in range(46)]
        rows += [row(f"a{i}", synth="multi_hop_abstract_query_synthesizer") for i in range(33)]
        rows += [row(f"m{i}", synth="multi_hop_specific_query_synthesizer") for i in range(21)]

        got = stratified_sample(rows, 10)

        assert len({r["synthesizer"] for r in got}) == 3

    def test_returns_everything_when_the_sample_is_not_smaller(self):
        rows = [row("q1"), row("q2")]

        assert stratified_sample(rows, 5) == rows

    def test_is_deterministic(self):
        rows = [row(f"q{i}") for i in range(20)]

        assert stratified_sample(rows, 5) == stratified_sample(rows, 5)


class TestRunEval:
    def _fake_result(self, n=1):
        result = MagicMock()
        result.to_pandas.return_value = pd.DataFrame(
            [{"faithfulness": 0.9, "answer_relevancy": 0.8}] * n
        )
        return result

    def test_reports_means_and_writes_a_result_file(self, tmp_path):
        pipeline = MagicMock()
        pipeline.answer.return_value = {"answer": "a", "contexts": ["c"], "chunk_ids": [1]}

        with (
            patch("eval.run.RESULTS_DIR", tmp_path / "results"),
            patch("eval.run.EvaluationDataset"),
            patch("eval.run.evaluate", return_value=self._fake_result()),
        ):
            out = run_eval(pipeline, [row("q1")], MagicMock(), MagicMock())

        assert out["means"]["faithfulness"] == 0.9
        assert len(list((tmp_path / "results").glob("*.json"))) == 1

    def test_reports_the_answering_subset_beside_the_full_mean(self, tmp_path):
        """A correct abstention scoring 0 on relevancy should be visible as an
        abstention, not folded into the headline number."""
        pipeline = MagicMock()
        pipeline.answer.side_effect = [
            {"answer": "a real answer", "contexts": ["c"], "chunk_ids": [1]},
            {"answer": "the context does not contain that", "contexts": ["c"], "chunk_ids": [1]},
        ]

        with (
            patch("eval.run.RESULTS_DIR", tmp_path / "results"),
            patch("eval.run.EvaluationDataset"),
            patch("eval.run.evaluate", return_value=self._fake_result(2)),
        ):
            out = run_eval(pipeline, [row("q1"), row("q2")], MagicMock(), MagicMock())

        assert out["n_abstentions"] == 1 and out["means_excluding_abstentions"]
