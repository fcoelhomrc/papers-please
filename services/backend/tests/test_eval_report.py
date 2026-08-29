"""Unit test for eval/report.py's markdown report generation.

Doesn't import ragas - no isolation needed.
"""
from unittest.mock import patch

from eval.report import write_markdown_report


def test_writes_markdown_with_summary_and_per_question_tables(tmp_path):
    dataset_rows = [
        {"category": "grounded", "domain": "robotics", "question": "What robot?", "ground_truth": "iCub"},
        {"category": "edge_case", "subtype": "off_topic_no_match", "question": "Off topic?", "ground_truth": "No"},
    ]
    output = {
        "variant": "fixed",
        "means": {"faithfulness": 0.9, "answer_relevancy": 0.75},
        "per_question": [
            {"user_input": "What robot?", "faithfulness": 1.0, "answer_relevancy": 0.9},
            {"user_input": "Off topic?", "faithfulness": 0.8, "answer_relevancy": 0.6},
        ],
    }

    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "claude-haiku-4-5", "claude-haiku-4-5")

    assert path.exists()
    text = path.read_text()

    # header + methodology
    assert "# Eval report — `fixed` pipeline" in text
    assert "claude-haiku-4-5" in text
    assert "manual only" in text.lower()

    # summary table has the mean scores
    assert "| faithfulness | 0.900 |" in text
    assert "| answer_relevancy | 0.750 |" in text

    # per-question table has both questions with their real scores
    assert "What robot?" in text
    assert "1.000" in text  # that question's faithfulness
    assert "Off topic?" in text
    assert "0.600" in text  # that question's answer_relevancy

    # category breakdown present
    assert "grounded" in text
    assert "edge_case" in text


def test_records_prompt_version_and_retrieval_metadata(tmp_path):
    """A report has to name the prompt that produced its score - that's the
    whole point of versioning them."""
    dataset_rows = [{"category": "grounded", "question": "q", "ground_truth": "a"}]
    output = {
        "variant": "agentic",
        "means": {"faithfulness": 0.5},
        "per_question": [{"user_input": "q", "faithfulness": 0.5}],
        "prompt_versions": {"orchestrator": "v2", "fixed_rag": "v1"},
        "retrieval": {
            "embed_model": "BAAI/bge-small-en-v1.5",
            "query_prompt": "Represent this sentence for searching relevant passages: ",
        },
    }

    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "claude-haiku-4-5", "claude-haiku-4-5")

    text = path.read_text()
    assert "`prompts/orchestrator/v2.md`" in text
    # an agentic run never loads the fixed_rag prompt - crediting its score to
    # a prompt that didn't run would be worse than omitting it
    assert "fixed_rag" not in text
    assert "BAAI/bge-small-en-v1.5" in text


def test_fixed_variant_names_the_fixed_prompt(tmp_path):
    dataset_rows = [{"category": "grounded", "question": "q", "ground_truth": "a"}]
    output = {
        "variant": "fixed",
        "means": {"faithfulness": 0.9},
        "per_question": [{"user_input": "q", "faithfulness": 0.9}],
        "prompt_versions": {"orchestrator": "v1", "fixed_rag": "v1"},
    }

    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "m", "m")

    text = path.read_text()
    assert "`prompts/fixed_rag/v1.md`" in text
    assert "orchestrator" not in text


def test_omits_prompt_line_for_older_results_without_versions(tmp_path):
    """Results JSON written before this feature has no prompt_versions key -
    rendering those must not crash or invent a version."""
    dataset_rows = [{"category": "grounded", "question": "q", "ground_truth": "a"}]
    output = {
        "variant": "fixed",
        "means": {"faithfulness": 0.9},
        "per_question": [{"user_input": "q", "faithfulness": 0.9}],
    }

    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "m", "m")

    assert "**Prompt**" not in path.read_text()
