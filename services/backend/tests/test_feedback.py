"""Tests for feedback capture and its route into the eval set (#34).

The API tests run against a stubbed session (no Postgres); the proposal
tests run against fake rows. Integration coverage against a real database
lives in test_documents_filter_integration.py's fixtures if it's ever needed
- the query here is a single filtered select, not the kind of SQL that
surprises you.
"""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api
from eval.feedback import collect, to_dataset_rows
from schemas import FeedbackRequest


@pytest.fixture
def client():
    return TestClient(api.app)


class TestFeedbackRequestValidation:
    def test_a_verdict_must_be_up_or_down(self):
        with pytest.raises(ValueError):
            FeedbackRequest(query="q", verdict="maybe")

    def test_a_query_is_required(self):
        """A judgement with no question attached cannot become an eval row,
        which is the entire reason for collecting these."""
        with pytest.raises(ValueError):
            FeedbackRequest(query="", verdict="up")

    def test_kind_is_constrained(self):
        with pytest.raises(ValueError):
            FeedbackRequest(query="q", verdict="up", kind="freeform")

    def test_defaults_to_a_search_judgement(self):
        assert FeedbackRequest(query="q", verdict="up").kind == "search"


class TestFeedbackEndpoints:
    def test_posting_feedback_persists_the_judgement(self, client):
        captured = []

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: None

        def add(row):
            captured.append(row)
            row.id = 1
            row.created_at = datetime.now(timezone.utc)

        session.add.side_effect = add
        api.app.dependency_overrides[api.get_engine] = lambda: MagicMock()
        try:
            with patch("api.Session", return_value=session):
                r = client.post(
                    "/feedback",
                    json={
                        "kind": "search",
                        "query": "fall recovery",
                        "doc_id": 3,
                        "chunk_id": 118,
                        "verdict": "up",
                    },
                )
        finally:
            api.app.dependency_overrides.clear()

        assert r.status_code == 200
        assert captured[0].query == "fall recovery"
        assert captured[0].doc_id == 3
        assert captured[0].verdict == "up"

    def test_an_invalid_verdict_is_rejected_before_the_database(self, client):
        api.app.dependency_overrides[api.get_engine] = lambda: MagicMock()
        try:
            r = client.post("/feedback", json={"query": "q", "verdict": "sideways"})
        finally:
            api.app.dependency_overrides.clear()

        assert r.status_code == 422


def _row(query, doc_id, verdict="up", kind="search"):
    return SimpleNamespace(
        query=query, doc_id=doc_id, verdict=verdict, kind=kind, chunk_id=None
    )


def _session_with(rows, docs):
    """A session returning `rows` from the feedback select and `docs` from
    the document lookup, in the order collect() issues them."""
    session = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = rows
    feedback_result = MagicMock()
    feedback_result.scalars.return_value = scalars
    doc_result = MagicMock()
    doc_result.all.return_value = list(docs.items())
    session.execute.side_effect = [feedback_result, doc_result]
    return session


class TestCollect:
    def test_groups_one_proposal_per_question(self):
        """Two searches of the same question with different results must
        contribute both papers to one row, not two rows disagreeing about
        what is relevant to the same question."""
        session = _session_with(
            [_row("fall recovery", 3), _row("Fall Recovery", 7)],
            {3: "src-3", 7: "src-7"},
        )

        [proposal] = collect(session)

        assert proposal["question"] == "fall recovery"
        assert proposal["relevant_source_ids"] == ["src-3", "src-7"]
        assert proposal["votes"] == 2

    def test_the_same_paper_twice_is_one_label(self):
        session = _session_with(
            [_row("q", 3), _row("q", 3)], {3: "src-3"}
        )

        [proposal] = collect(session)

        assert proposal["relevant_source_ids"] == ["src-3"]

    def test_min_votes_filters_out_one_off_clicks(self):
        session = _session_with([_row("q", 3)], {3: "src-3"})

        assert collect(session, min_votes=2) == []

    def test_a_question_with_no_resolvable_document_is_dropped(self):
        """A judgement about a document that no longer exists cannot become
        a `relevant_source_ids` entry."""
        session = _session_with([_row("q", 999)], {})

        assert collect(session) == []


class TestToDatasetRows:
    def _proposal(self, question="q", ids=("src-3",)):
        return {
            "question": question,
            "relevant_source_ids": list(ids),
            "votes": 1,
            "kinds": ["search"],
        }

    def test_ground_truth_is_left_blank_for_a_human(self):
        """A thumb says which paper is relevant, not what the answer is.
        Inventing one would poison every faithfulness score computed against
        it thereafter."""
        [row] = to_dataset_rows([self._proposal()], skip=set())

        assert row["ground_truth"] == ""
        assert row["relevant_source_ids"] == ["src-3"]

    def test_carries_the_fields_stratified_sampling_groups_on(self):
        """eval/run.py:stratified_sample keys on category x domain; a row
        without them lands in an unnamed stratum."""
        [row] = to_dataset_rows([self._proposal()], skip=set())

        assert row["category"] == "grounded"
        assert row["domain"] == "feedback"

    def test_questions_already_in_the_dataset_are_skipped(self):
        rows = to_dataset_rows(
            [self._proposal("Already Asked")], skip={"already asked"}
        )

        assert rows == []

    def test_output_is_valid_jsonl(self):
        rows = to_dataset_rows([self._proposal("a"), self._proposal("b")], skip=set())

        for row in rows:
            assert json.loads(json.dumps(row))["question"] in {"a", "b"}
