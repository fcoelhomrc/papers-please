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

        records, retrieved, _ = answer_all(pipeline, [row("q0001", "why?")])

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

        records, *_ = answer_all(pipeline, [row("q0001")])

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

        records, retrieved, _ = answer_all(
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


class TestEmptyAnswerGuard:
    """An empty answer looks like a successful run right up until the metrics
    come back nan. It cost a full judged run to find out once; the guard makes
    it fail at the point it happens, before any judge call is paid for."""

    def test_refuses_to_judge_blank_answers(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = {"answer": "", "contexts": ["c"], "chunk_ids": [1]}

        with pytest.raises(RuntimeError, match="came back empty"):
            answer_all(pipeline, [row("q1")])

    def test_whitespace_only_counts_as_empty(self):
        pipeline = MagicMock()
        pipeline.answer.return_value = {"answer": "   \n", "contexts": ["c"], "chunk_ids": [1]}

        with pytest.raises(RuntimeError, match="came back empty"):
            answer_all(pipeline, [row("q1")])

    def test_a_recorded_pipeline_failure_is_not_treated_as_empty(self):
        """The error text is a real response for judging purposes - it is the
        run telling you what happened, not a blank."""
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("boom")

        records, *_ = answer_all(pipeline, [row("q1")])

        assert "boom" in records[0]["response"]


class TestNetworkLossGuard:
    """A run that loses its connection partway through still completes: every
    question after the drop records its exception as the answer, the judge
    scores those strings, and the result file looks structurally valid. It
    would then win judged_by_arm()'s newest-run-per-arm and silently replace a
    good run in the figures."""

    def test_refuses_when_most_questions_failed(self):
        pipeline = MagicMock()
        pipeline.answer.side_effect = RuntimeError("connection reset")

        with pytest.raises(RuntimeError, match="refusing to write"):
            answer_all(pipeline, [row(f"q{i}") for i in range(20)])

    def test_a_couple_of_failures_still_completes(self):
        """Per-question resilience is the point - one bad question must not
        discard the other 99 already paid for."""
        pipeline = MagicMock()
        pipeline.answer.side_effect = (
            [RuntimeError("blip")]
            + [{"answer": "a", "contexts": ["c"], "chunk_ids": [1]}] * 19
        )

        records, _, failed = answer_all(pipeline, [row(f"q{i}") for i in range(20)])

        assert len(records) == 20 and failed == 1


class TestCoverage:
    """A judge call that runs out of output budget is recorded by ragas as
    NaN, and pandas' .mean() skips NaN - so a metric scored over half the set
    reads as a clean number with nothing to say it was halved. An audit found
    faithfulness nan on 51 of 100 questions in a run whose headline figure had
    already been reported."""

    def test_reports_how_many_questions_each_metric_scored(self, tmp_path):
        import math

        pipeline = MagicMock()
        pipeline.answer.return_value = {"answer": "a", "contexts": ["c"], "chunk_ids": [1]}
        result = MagicMock()
        result.to_pandas.return_value = pd.DataFrame([
            {"faithfulness": 0.9, "answer_relevancy": 0.8},
            {"faithfulness": math.nan, "answer_relevancy": 0.7},
        ])

        with (
            patch("eval.run.RESULTS_DIR", tmp_path / "results"),
            patch("eval.run.EvaluationDataset"),
            patch("eval.run.evaluate", return_value=result),
        ):
            out = run_eval(pipeline, [row("q1"), row("q2")], MagicMock(), MagicMock())

        assert out["coverage"] == {"faithfulness": 1, "answer_relevancy": 2}


class TestArmModeGuard:
    """`--mode bm25 --arm hyde` used to run a dense search and file it as BM25.

    The guard has to fire before the run does any work: a judged run costs
    ~$0.19 of judge tokens, and a mislabelled result is worse than a crash
    because it survives into the figures.
    """

    def _main(self, monkeypatch, argv, *, spent):
        import eval.review
        from eval import run as run_mod

        monkeypatch.setattr("sys.argv", ["eval.run", *argv])
        # Anything past the guard would begin the actual run.
        monkeypatch.setattr(eval.review, "load_curated",
                            lambda *a, **k: spent.append("loaded") or [])
        monkeypatch.setattr(run_mod, "build_pipeline",
                            lambda *a, **k: spent.append("built") or (None, "m"))
        return run_mod.main

    def test_rejects_hyde_under_bm25(self, monkeypatch):
        spent = []
        main = self._main(monkeypatch, ["--mode", "bm25", "--arm", "hyde"], spent=spent)
        with pytest.raises(SystemExit) as e:
            main()
        assert "hyde" in str(e.value) and "bm25" in str(e.value)
        # The supported mode is named, so the message is actionable.
        assert "semantic" in str(e.value)
        assert spent == []

    def test_rejects_an_arm_under_a_hybrid_mode(self, monkeypatch):
        spent = []
        main = self._main(monkeypatch, ["--mode", "hybrid", "--arm", "multi_query"],
                          spent=spent)
        with pytest.raises(SystemExit):
            main()
        assert spent == []

    def test_allows_hyde_under_semantic(self, monkeypatch):
        """The guard must not block the 15 dense arm runs that remain."""
        spent = []
        main = self._main(monkeypatch, ["--mode", "semantic", "--arm", "hyde"],
                          spent=spent)
        with pytest.raises(Exception) as e:
            main()
        # It got past the guard - whatever it failed on next, it was not this.
        assert not isinstance(e.value, SystemExit) or "cannot run in mode" not in str(e.value)

    def test_allows_multi_query_under_bm25(self, monkeypatch):
        spent = []
        main = self._main(monkeypatch, ["--mode", "bm25", "--arm", "multi_query"],
                          spent=spent)
        with pytest.raises(Exception) as e:
            main()
        assert not isinstance(e.value, SystemExit) or "cannot run in mode" not in str(e.value)
