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


class TestQuestionReview:
    """Review decisions live in a JSON sidecar, not in generated.jsonl, so
    the generator's output stays immutable and a diff shows exactly what a
    human changed."""

    @pytest.fixture
    def paths(self, tmp_path, monkeypatch):
        import json

        import eval.review as review

        gen = tmp_path / "generated.jsonl"
        gen.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"q{i:04d}",
                        "question": f"q{i}?",
                        "reference": f"a{i}",
                        "reference_contexts": ["ctx"],
                        "reference_chunk_ids": [i],
                        "reference_doc_ids": [i],
                        "topics": ["agents" if i else "retrieval"],
                        "synthesizer": "single_hop_specific_query_synthesizer",
                    }
                )
                for i in range(3)
            )
        )
        monkeypatch.setattr(review, "GENERATED_PATH", gen)
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "review.json")
        return review

    def test_starts_undecided(self, paths):
        assert {q["decision"] for q in paths.questions()} == {"undecided"}

    def test_keep_is_recorded_and_merged_back(self, paths):
        paths.review_question("q0000", decision="keep")

        assert next(q for q in paths.questions() if q["id"] == "q0000")["decision"] == "keep"

    def test_generated_file_is_never_rewritten(self, paths):
        before = paths.GENERATED_PATH.read_text()
        paths.review_question("q0000", decision="drop", question="reworded?")

        assert paths.GENERATED_PATH.read_text() == before

    def test_edit_keeps_the_original_alongside(self, paths):
        """So the UI can show what changed rather than quietly replacing it."""
        paths.review_question("q0001", question="reworded?")
        q = next(q for q in paths.questions() if q["id"] == "q0001")

        assert (q["question"], q["original_question"], q["edited"]) == (
            "reworded?",
            "q1?",
            True,
        )

    def test_decisions_survive_a_reload(self, paths):
        paths.review_question("q0002", decision="keep", note="good one")
        state = paths.load_review()

        assert state["q0002"] == {"decision": "keep", "note": "good one"}

    def test_unknown_question_id_raises(self, paths):
        with pytest.raises(ValueError, match="no generated question"):
            paths.review_question("q9999", decision="keep")

    def test_unknown_decision_raises(self, paths):
        with pytest.raises(ValueError, match="unknown decision"):
            paths.review_question("q0000", decision="maybe")

    def test_summary_counts_only_kept_by_topic(self, paths):
        """The topic counters are what rebalances a skewed set - agents took
        75 of 131 generated questions - so they must reflect keeps, not the
        whole generated set."""
        paths.review_question("q0000", decision="keep")
        paths.review_question("q0001", decision="drop")
        summary = paths.question_summary(paths.questions())

        assert summary["kept_by_topic"] == {"retrieval": 1}
        assert (summary["kept"], summary["dropped"], summary["undecided"]) == (1, 1, 1)


class TestCuratedExport:
    """The harness reads a resolved file, not generated.jsonl plus a review
    sidecar, so a review still in progress cannot change what an experiment
    measured halfway through it."""

    @pytest.fixture
    def paths(self, tmp_path, monkeypatch):
        import json

        import eval.review as review

        gen = tmp_path / "generated.jsonl"
        gen.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"q{i:04d}",
                        "question": f"q{i}?",
                        "reference": f"a{i}",
                        "reference_contexts": ["ctx"],
                        "reference_chunk_ids": [i],
                        "reference_doc_ids": [i],
                        "topics": ["agents"],
                        "synthesizer": "single_hop_specific_query_synthesizer",
                    }
                )
                for i in range(3)
            )
        )
        monkeypatch.setattr(review, "GENERATED_PATH", gen)
        monkeypatch.setattr(review, "REVIEW_PATH", tmp_path / "review.json")
        monkeypatch.setattr(review, "CURATED_PATH", tmp_path / "curated.jsonl")
        return review

    def test_exports_only_kept_questions(self, paths):
        paths.review_question("q0000", decision="keep")
        paths.review_question("q0001", decision="drop")

        assert [q["id"] for q in paths.curated()] == ["q0000"]

    def test_applies_edits(self, paths):
        paths.review_question("q0002", decision="keep", question="reworded?")

        assert paths.curated()[0]["question"] == "reworded?"

    def test_round_trips_through_disk(self, paths):
        paths.review_question("q0000", decision="keep")
        paths.write_curated()

        assert paths.load_curated()[0]["id"] == "q0000"

    def test_missing_file_says_how_to_produce_it(self, paths):
        with pytest.raises(FileNotFoundError, match="eval.review export"):
            paths.load_curated()
