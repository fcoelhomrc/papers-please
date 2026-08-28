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
