"""Corpus curation: promoting staged candidates into the eval corpus.

State lives in `documents.corpus` rather than a side file, because that is
already the column every retrieval path filters on - a second source of truth
would let the review UI and the retriever disagree about what the corpus is.
"""
from unittest.mock import MagicMock, patch

import pytest
from eval.review import CANDIDATE, EVAL, REJECTED, decide, topic_summary
from fastapi.testclient import TestClient


class TestDecide:
    def _session(self, found=True):
        session = MagicMock()
        session.execute.return_value.first.return_value = (1,) if found else None
        return session

    def test_keep_promotes_into_the_eval_corpus(self):
        session = self._session()

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            assert decide(1, "keep") == EVAL

    def test_reject_returns_the_paper_to_the_main_corpus(self):
        """Rejecting is not a delete: the paper stays fetched so re-staging
        does not re-offer it, and the decision survives."""
        session = self._session()

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            assert decide(1, "reject") == REJECTED

    def test_reset_returns_a_paper_to_the_candidate_pool(self):
        """Curation is a long manual pass; a misclick must not be one-way.
        Without reset, a rejected paper drops out of the list and the only
        route back is a hand-written UPDATE."""
        session = self._session()

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            assert decide(1, "reset") == CANDIDATE

    def test_unknown_decision_raises(self):
        """A typo must not silently move a paper into the eval corpus."""
        with pytest.raises(ValueError, match="unknown decision"):
            decide(1, "maybe")

    def test_a_document_outside_any_topic_raises(self):
        """The update is scoped to topic-tagged rows, so an arbitrary doc_id
        cannot be swept into the corpus by a stray request."""
        session = self._session(found=False)

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            with pytest.raises(ValueError, match="no staged candidate"):
                decide(999, "keep")


class TestTopicSummary:
    def test_counts_each_state_separately(self):
        session = MagicMock()
        session.execute.return_value.all.return_value = [
            ("retrieval", CANDIDATE, 18),
            ("retrieval", EVAL, 20),
            ("retrieval", REJECTED, 2),
        ]

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            summary = topic_summary()

        assert summary == [
            {"topic": "retrieval", "candidates": 18, "kept": 20, "rejected": 2}
        ]

    def test_reports_zero_for_states_with_no_rows(self):
        """A topic nobody has started on must still appear, or it silently
        drops out of the counters the reviewer is steering by."""
        session = MagicMock()
        session.execute.return_value.all.return_value = [("agents", CANDIDATE, 40)]

        with patch("eval.review.Session") as sess, patch("eval.review.PostgresInterface"):
            sess.return_value.__enter__.return_value = session
            assert topic_summary()[0]["kept"] == 0


class TestEndpoints:
    @pytest.fixture
    def client(self):
        import api

        return TestClient(api.app)

    def test_list_returns_targets_topics_and_candidates(self, client):
        with (
            patch("eval.review.candidates", return_value=[]),
            patch("eval.review.topic_summary", return_value=[]),
        ):
            body = client.get("/eval/candidates").json()

        assert body["target_per_topic"] == 20

    def test_list_forwards_the_topic_filter(self, client):
        with (
            patch("eval.review.candidates", return_value=[]) as cands,
            patch("eval.review.topic_summary", return_value=[]),
        ):
            client.get("/eval/candidates?topic=agents")

        assert cands.call_args.args[0] == "agents"

    def test_patch_returns_the_new_corpus(self, client):
        with patch("eval.review.decide", return_value="eval"):
            body = client.patch("/eval/candidates/7", json={"decision": "keep"}).json()

        assert body == {"id": 7, "corpus": "eval"}

    def test_patch_accepts_reset(self, client):
        with patch("eval.review.decide", return_value="candidate") as d:
            client.patch("/eval/candidates/7", json={"decision": "reset"})

        assert d.call_args.args[1] == "reset"

    def test_patch_rejects_an_unknown_decision_before_the_database(self, client):
        res = client.patch("/eval/candidates/7", json={"decision": "sideways"})

        assert res.status_code == 422

    def test_patch_on_an_unknown_document_is_a_404(self, client):
        with patch("eval.review.decide", side_effect=ValueError("no staged candidate")):
            res = client.patch("/eval/candidates/999", json={"decision": "keep"})

        assert res.status_code == 404
