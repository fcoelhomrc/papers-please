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


def test_includes_judge_free_retrieval_section(tmp_path):
    """Retrieval metrics separate 'never retrieved it' from 'retrieved it and
    the model ignored it' - the judged metrics alone can't tell those apart."""
    dataset_rows = [{"category": "grounded", "question": "q", "ground_truth": "a"}]
    output = {
        "variant": "agentic",
        "means": {"faithfulness": 0.5},
        "per_question": [{"user_input": "q", "faithfulness": 0.5}],
        "retrieval_metrics": {
            "recall": 0.81, "precision": 0.2, "hit_rate": 0.9, "mrr": 0.75,
            "ndcg": 0.7, "n_retrieval": 42, "n_abstention": 8,
            "abstention_precision": 0.5, "mean_false_positives": 1.2,
        },
    }

    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "m", "m")

    text = path.read_text()
    assert "Retrieval (no judge involved)" in text
    assert "0.810" in text and "ndcg" in text
    assert "abstention_precision" in text


def test_omits_retrieval_section_when_unlabeled(tmp_path):
    dataset_rows = [{"category": "grounded", "question": "q", "ground_truth": "a"}]
    output = {
        "variant": "fixed",
        "means": {"faithfulness": 0.9},
        "per_question": [{"user_input": "q", "faithfulness": 0.9}],
        "retrieval_metrics": {},
    }
    with patch("eval.report.REPORTS_DIR", tmp_path):
        path = write_markdown_report(output, dataset_rows, "m", "m")
    assert "Retrieval (no judge involved)" not in path.read_text()


class TestJudgeSpendReporting:
    """#26 — a score with no cost next to it is how a 50-question run quietly
    grows into a 1M-token one. The report states spend; the ledger parses it
    back out."""

    def test_report_states_judge_spend(self, tmp_path):
        from unittest.mock import patch

        from eval.report import write_markdown_report

        output = {
            "variant": "fixed",
            "means": {"faithfulness": 0.8},
            "per_question": [{"user_input": "q1", "faithfulness": 0.8}],
            "n_questions": 1,
            "n_dataset": 1,
            "judge_spend": {"input_tokens": 89412, "output_tokens": 12004, "usd": 0.1494},
        }
        rows = [{"category": "grounded", "domain": "robotics", "question": "q1"}]

        with patch("eval.report.REPORTS_DIR", tmp_path):
            text = write_markdown_report(output, rows, "m", "j").read_text()

        assert "**Judge spend**: 89,412 in / 12,004 out tokens — $0.1494" in text

    def test_report_flags_a_sampled_run(self, tmp_path):
        from unittest.mock import patch

        from eval.report import write_markdown_report

        output = {
            "variant": "fixed",
            "means": {"faithfulness": 0.8},
            "per_question": [{"user_input": "q1", "faithfulness": 0.8}],
            "n_questions": 1,
            "n_dataset": 50,
        }
        rows = [{"category": "grounded", "domain": "robotics", "question": "q1"}]

        with patch("eval.report.REPORTS_DIR", tmp_path):
            text = write_markdown_report(output, rows, "m", "j").read_text()

        assert "1 of 50 questions" in text
        assert "stratified sample" in text

    def test_report_omits_spend_when_it_was_not_recorded(self, tmp_path):
        from unittest.mock import patch

        from eval.report import write_markdown_report

        output = {
            "variant": "fixed",
            "means": {"faithfulness": 0.8},
            "per_question": [{"user_input": "q1", "faithfulness": 0.8}],
        }
        rows = [{"category": "grounded", "domain": "robotics", "question": "q1"}]

        with patch("eval.report.REPORTS_DIR", tmp_path):
            text = write_markdown_report(output, rows, "m", "j").read_text()

        assert "Judge spend" not in text
        assert "1 questions" in text  # no "of N" when nothing was sampled

    def test_ledger_round_trips_spend_and_real_question_count(self, tmp_path):
        """The report is the only surviving record of the early judged runs,
        so whatever it states has to be parseable back out."""
        from unittest.mock import patch

        from eval.ingest import parse_judged_report
        from eval.report import write_markdown_report

        output = {
            "variant": "agentic",
            "means": {"faithfulness": 0.5, "answer_relevancy": 0.7},
            "per_question": [{"user_input": "q1", "faithfulness": 0.5}],
            "n_questions": 1,
            "n_dataset": 50,
            "judge_spend": {"input_tokens": 1234, "output_tokens": 56, "usd": 0.0043},
        }
        rows = [{"category": "grounded", "domain": "robotics", "question": "q1"}]

        with patch("eval.report.REPORTS_DIR", tmp_path):
            path = write_markdown_report(output, rows, "claude-haiku-4-5", "claude-haiku-4-5")

        record = parse_judged_report(path)

        assert record["n_questions"] == 1
        assert record["judge_spend"] == {
            "input_tokens": 1234,
            "output_tokens": 56,
            "usd": 0.0043,
        }
        assert record["metrics"] == {"faithfulness": 0.5, "answer_relevancy": 0.7}

    def test_old_reports_without_spend_still_parse(self, tmp_path):
        """Every committed report predates #26. Backfilling the ledger from
        them must keep working, spend simply absent."""
        from eval.ingest import parse_judged_report

        path = tmp_path / "fixed-20260828T152436Z.md"
        path.write_text(
            "# Eval report — `fixed` pipeline\n\n"
            "- **Run at (UTC)**: 2026-08-28T15:24:36+00:00\n"
            "- **Pipeline model**: `claude-haiku-4-5`\n"
            "- **Judge model**: `claude-haiku-4-5` (via Ragas)\n"
            "- **Dataset**: `eval/dataset.jsonl` — 50 questions (48 grounded, 2 edge case)\n"
            "\n## Summary (mean across all questions)\n\n"
            "| Metric | Mean score |\n|---|---|\n| faithfulness | 0.829 |\n"
        )

        record = parse_judged_report(path)

        assert record["n_questions"] == 50
        assert record["judge_spend"] == {}
        assert record["metrics"] == {"faithfulness": 0.829}
