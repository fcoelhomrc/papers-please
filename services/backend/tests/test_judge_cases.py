"""Tests for the judge-comparison case set and its human labels (#18 follow-up)."""
import json

from eval.judge_cases import (
    CASES,
    agreement_with_proposed,
    as_records,
    load_labels,
    summary,
)


class TestCaseSet:
    def test_every_statement_is_unique(self):
        """A duplicate would be labelled twice and double-count in kappa."""
        statements = [s for _, s, _, _ in CASES]

        assert len(statements) == len(set(statements))

    def test_labels_are_binary(self):
        assert {label for _, _, label, _ in CASES} == {0, 1}

    def test_classes_are_roughly_balanced(self):
        """Cohen's kappa becomes unstable when one class dominates, and a
        judge could score well on a skewed set by always answering the
        majority class."""
        positive = sum(1 for _, _, label, _ in CASES if label == 1)

        assert 0.4 <= positive / len(CASES) <= 0.6

    def test_includes_the_discriminating_case_types(self):
        """Obvious supported/unrelated pairs decided nothing last time — four
        models scored identically. These are the types that separate."""
        assert {
            "corrupted_number",
            "corrupted_entity",
            "true_but_absent",
            "overgeneralised",
            "paraphrase",
        } <= set(summary())

    def test_ids_are_stable_and_unique(self):
        records = as_records()

        assert len({r["id"] for r in records}) == len(records)
        assert as_records()[0]["id"] == records[0]["id"]


class TestLabelLoading:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_labels(tmp_path / "nope.json") == {}

    def test_unsure_labels_are_dropped_not_coerced(self, tmp_path):
        """An item nobody could decide is not ground truth. Guessing would
        put noise into the numbers this exists to make trustworthy."""
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"labels": [
            {"id": "c000", "label": 1},
            {"id": "c001", "label": None},
            {"id": "c002", "label": 0},
        ]}))

        assert load_labels(path) == {"c000": 1, "c002": 0}

    def test_agreement_reports_where_construction_was_wrong(self):
        """Disagreement means the cases were sloppier than claimed — the fix
        is the cases, not quietly keeping my own labels."""
        proposed = {r["id"]: r["proposed_label"] for r in as_records()}
        first, second = list(proposed)[:2]
        labels = {first: proposed[first], second: 1 - proposed[second]}

        result = agreement_with_proposed(labels)

        assert result["n_labelled"] == 2
        assert result["n_disagreed"] == 1
        assert result["disagreed"][0]["id"] == second
        assert "case_type" in result["disagreed"][0]

    def test_perfect_agreement_reports_zero_rate(self):
        labels = {r["id"]: r["proposed_label"] for r in as_records()}

        assert agreement_with_proposed(labels)["rate"] == 0.0
