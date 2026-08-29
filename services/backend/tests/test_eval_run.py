"""Unit tests for eval/run.py's harness wiring - ragas.evaluate() itself is
mocked (a real call costs real LLM judge tokens per metric per question,
not something to do on every test run).

Marked `eval` and excluded from default collection entirely (not just
marker-deselected - see pyproject.toml's --ignore): importing ragas
triggers nest_asyncio.apply() at import time (ragas/executor.py,
unconditional), which poisons the process-wide asyncio event loop and
breaks FastAPI TestClient (anyio-based) tests if both run in the same
pytest process - and pytest imports every collected file to read its
markers, so marker-based deselection alone isn't enough; the file has to
be kept out of collection too. Run in isolation with:

    uv run pytest -o addopts="" tests/test_eval_run.py -m eval
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eval.run import load_dataset, run_eval

pytestmark = pytest.mark.eval


def test_load_dataset_parses_jsonl(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text('{"question": "q1", "ground_truth": "a1"}\n{"question": "q2", "ground_truth": "a2"}\n')

    rows = load_dataset(path)

    assert rows == [
        {"question": "q1", "ground_truth": "a1"},
        {"question": "q2", "ground_truth": "a2"},
    ]


def test_run_eval_shapes_pipeline_output_into_ragas_records(tmp_path):
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text('{"question": "q1", "ground_truth": "a1"}\n')

    pipeline = MagicMock()
    pipeline.answer.return_value = {"answer": "the answer", "contexts": ["ctx1", "ctx2"]}

    fake_df = pd.DataFrame([{"faithfulness": 0.9, "answer_relevancy": 0.8}])
    fake_eval_result = MagicMock()
    fake_eval_result.to_pandas.return_value = fake_df

    with (
        patch("eval.run.RESULTS_DIR", tmp_path / "results"),
        patch("eval.run.EvaluationDataset") as MockDataset,
        patch("eval.run.evaluate", return_value=fake_eval_result) as mock_evaluate,
        patch("eval.run.write_markdown_report", return_value=tmp_path / "report.md") as mock_report,
    ):
        output = run_eval(pipeline, dataset_path, "fixed", judge_llm=MagicMock(), judge_embeddings=MagicMock())

    pipeline.answer.assert_called_once_with("q1")
    records_passed = MockDataset.from_list.call_args.args[0]
    assert records_passed == [
        {
            "user_input": "q1",
            "response": "the answer",
            "retrieved_contexts": ["ctx1", "ctx2"],
            "reference": "a1",
        }
    ]
    mock_evaluate.assert_called_once()
    assert output["variant"] == "fixed"
    assert output["means"]["faithfulness"] == 0.9
    assert output["report_path"] == str(tmp_path / "report.md")
    mock_report.assert_called_once()

    # a results file was actually written, to the patched (tmp) dir, not the real one
    result_files = list((tmp_path / "results").glob("*.json"))
    assert len(result_files) == 1


def test_run_eval_substitutes_placeholder_when_no_context_retrieved(tmp_path):
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text('{"question": "off-topic question", "ground_truth": "no"}\n')

    pipeline = MagicMock()
    pipeline.answer.return_value = {"answer": "nothing relevant found", "contexts": []}

    fake_eval_result = MagicMock()
    fake_eval_result.to_pandas.return_value = pd.DataFrame([{"faithfulness": 1.0}])

    with (
        patch("eval.run.RESULTS_DIR", tmp_path / "results"),
        patch("eval.run.EvaluationDataset") as MockDataset,
        patch("eval.run.evaluate", return_value=fake_eval_result),
        patch("eval.run.write_markdown_report", return_value=tmp_path / "report.md"),
    ):
        run_eval(pipeline, dataset_path, "agentic", judge_llm=MagicMock(), judge_embeddings=MagicMock())

    records_passed = MockDataset.from_list.call_args.args[0]
    assert records_passed[0]["retrieved_contexts"] == ["(no context retrieved)"]


def test_run_eval_survives_one_question_failing(tmp_path):
    """Regression test for a real incident: one question hit LangGraph's
    recursion limit (uncaught GraphRecursionError), which crashed the whole
    loop before anything reached disk - discarding ~40 other questions'
    worth of real, already-paid-for answers. One failure must not cost the
    rest."""
    dataset_path = tmp_path / "ds.jsonl"
    dataset_path.write_text(
        '{"question": "q1", "ground_truth": "a1"}\n'
        '{"question": "q2 (this one blows up)", "ground_truth": "a2"}\n'
        '{"question": "q3", "ground_truth": "a3"}\n'
    )

    pipeline = MagicMock()
    pipeline.answer.side_effect = [
        {"answer": "answer 1", "contexts": ["ctx1"]},
        RuntimeError("recursion limit hit"),
        {"answer": "answer 3", "contexts": ["ctx3"]},
    ]

    fake_eval_result = MagicMock()
    fake_eval_result.to_pandas.return_value = pd.DataFrame([{"faithfulness": 1.0}] * 3)

    with (
        patch("eval.run.RESULTS_DIR", tmp_path / "results"),
        patch("eval.run.EvaluationDataset") as MockDataset,
        patch("eval.run.evaluate", return_value=fake_eval_result),
        patch("eval.run.write_markdown_report", return_value=tmp_path / "report.md"),
    ):
        output = run_eval(pipeline, dataset_path, "agentic", judge_llm=MagicMock(), judge_embeddings=MagicMock())

    # all 3 questions made it into the ragas dataset - q2 as a recorded
    # failure, not silently dropped and not crashing the other two
    records_passed = MockDataset.from_list.call_args.args[0]
    assert len(records_passed) == 3
    assert records_passed[0]["response"] == "answer 1"
    assert "recursion limit hit" in records_passed[1]["response"]
    assert records_passed[1]["retrieved_contexts"] == ["(no context retrieved)"]
    assert records_passed[2]["response"] == "answer 3"

    # and the run still completed and wrote output, not aborted
    assert output["variant"] == "agentic"


class TestResolvePromptVersions:
    """`--prompt-version name=version` handling. Resolved before any API
    spend, so a typo fails immediately rather than after 50 paid questions."""

    def test_defaults_come_from_config(self):
        from eval.run import _resolve_prompt_versions

        assert _resolve_prompt_versions(None) == {"orchestrator": "v1", "fixed_rag": "v1"}

    def test_override_replaces_one_and_leaves_the_rest(self):
        from eval.run import _resolve_prompt_versions

        assert _resolve_prompt_versions(["orchestrator=v2"]) == {
            "orchestrator": "v2",
            "fixed_rag": "v1",
        }

    def test_multiple_overrides(self):
        from eval.run import _resolve_prompt_versions

        got = _resolve_prompt_versions(["orchestrator=v2", "fixed_rag=v3"])
        assert got == {"orchestrator": "v2", "fixed_rag": "v3"}

    def test_unknown_prompt_name_raises(self):
        from eval.run import _resolve_prompt_versions

        with pytest.raises(ValueError, match="unknown prompt"):
            _resolve_prompt_versions(["typoed_name=v2"])

    def test_malformed_override_raises(self):
        from eval.run import _resolve_prompt_versions

        with pytest.raises(ValueError, match="name=version"):
            _resolve_prompt_versions(["orchestrator"])
