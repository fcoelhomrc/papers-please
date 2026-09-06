"""Unit tests for eval/retrieval.py - pure functions, no infra."""
import math

import pytest

from eval.retrieval import (
    aggregate,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_question,
)


class TestRecall:
    def test_finds_all_relevant(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_finds_half(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, k=5) == 0.5

    def test_respects_k_cutoff(self):
        assert recall_at_k(["x", "x", "a"], {"a"}, k=2) == 0.0

    def test_duplicate_chunks_from_one_doc_count_once(self):
        """Retrieval returns chunks; several routinely share a document.
        Counting a doc twice would report recall above 1."""
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=5) == 0.5

    def test_undefined_without_relevant_docs(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), k=5)


class TestPrecision:
    def test_divides_by_k_not_by_results_returned(self):
        """Returning 1 correct result when 5 were asked for is not precision
        1.0 - it means the other 4 slots were wasted."""
        assert precision_at_k(["a"], {"a"}, k=5) == pytest.approx(0.2)

    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_zero_k_rejected(self):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k=0)


class TestHitRate:
    def test_one_relevant_is_enough(self):
        assert hit_rate_at_k(["x", "x", "a"], {"a"}, k=5) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x"], {"a"}, k=5) == 0.0

    def test_respects_cutoff(self):
        assert hit_rate_at_k(["x", "a"], {"a"}, k=1) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "x"], {"a"}) == 1.0

    def test_third_position(self):
        assert mrr(["x", "x", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found(self):
        assert mrr(["x"], {"a"}) == 0.0

    def test_uses_first_relevant_only(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_reversed_relevance_penalised(self):
        """Same documents found, worse order - must score lower."""
        good = ndcg_at_k(["a", "x"], {"a"}, k=2)
        bad = ndcg_at_k(["x", "a"], {"a"}, k=2)
        assert good > bad

    def test_credits_every_relevant_doc_not_just_the_first(self):
        """The difference from MRR, and what makes it right for the
        multi-document questions."""
        one = ndcg_at_k(["a", "x"], {"a", "b"}, k=2)
        both = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
        assert both > one

    def test_ideal_capped_by_k(self):
        """With 3 relevant docs but k=1, finding the top one is a perfect
        score for that budget - not 1/3."""
        assert ndcg_at_k(["a"], {"a", "b", "c"}, k=1) == pytest.approx(1.0)

    def test_discount_matches_definition(self):
        # single relevant doc at rank 2: dcg = 1/log2(3), idcg = 1/log2(2) = 1
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))

    def test_duplicates_do_not_inflate(self):
        assert ndcg_at_k(["a", "a"], {"a"}, k=2) == pytest.approx(1.0)

    def test_undefined_without_relevant_docs(self):
        with pytest.raises(ValueError):
            ndcg_at_k(["a"], set(), k=5)


class TestScoreQuestion:
    def test_retrieval_question_reports_all_metrics(self):
        got = score_question(["a", "x"], {"a"}, k=2)
        assert got["abstention"] is False
        assert set(got) >= {"recall", "precision", "hit_rate", "mrr", "ndcg"}

    def test_abstention_question_rewards_retrieving_nothing(self):
        assert score_question([], set(), k=5)["abstention_precision"] == 1.0

    def test_abstention_question_penalises_false_positives(self):
        got = score_question(["a", "b"], set(), k=5)
        assert got["abstention_precision"] == 0.0
        assert got["false_positives"] == 2

    def test_abstention_question_omits_undefined_metrics(self):
        """recall on a question with nothing to recall is not 0.0, it's
        meaningless - it must not appear."""
        got = score_question(["a"], set(), k=5)
        assert "recall" not in got and "ndcg" not in got


class TestAggregate:
    def test_averages_only_over_defined_questions(self):
        per_q = [
            score_question(["a"], {"a"}, k=1),      # recall 1.0
            score_question(["x"], {"a"}, k=1),      # recall 0.0
            score_question(["x"], set(), k=1),      # abstention - excluded
        ]
        out = aggregate(per_q)
        assert out["recall"] == 0.5  # not 1/3 - the abstention row is not a zero
        assert out["n_retrieval"] == 2
        assert out["n_abstention"] == 1

    def test_abstention_scored_separately(self):
        per_q = [
            score_question([], set(), k=5),
            score_question(["a", "b"], set(), k=5),
        ]
        out = aggregate(per_q)
        assert out["abstention_precision"] == 0.5
        assert out["mean_false_positives"] == 1.0

    def test_no_abstention_keys_when_no_such_questions(self):
        out = aggregate([score_question(["a"], {"a"}, k=1)])
        assert "abstention_precision" not in out


class TestRPrecision:
    """Precision@k is capped at min(R,k)/k, so on a set where most questions
    have one gold chunk precision@10 cannot exceed 0.1 - it measures the
    set's shape more than the retriever. R-precision moves the cutoff to R,
    so a perfect ranking scores 1.0 whatever R is."""

    def test_perfect_ranking_scores_one_whatever_r_is(self):
        from eval.retrieval import r_precision

        assert r_precision(["a", "b", "c"], {"a"}) == 1.0
        assert r_precision(["a", "b", "c"], {"a", "b"}) == 1.0
        assert r_precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0

    def test_cutoff_is_r_not_the_retrieved_length(self):
        """One relevant doc at rank 2 scores 0 when R=1: only rank 1 counts."""
        from eval.retrieval import r_precision

        assert r_precision(["x", "a"], {"a"}) == 0.0

    def test_partial_credit(self):
        from eval.retrieval import r_precision

        assert r_precision(["a", "x", "b"], {"a", "b"}) == 0.5

    def test_deduped_like_the_other_metrics(self):
        """Retrieval returns chunks; the same one twice must not score twice."""
        from eval.retrieval import r_precision

        assert r_precision(["a", "a"], {"a", "b"}) == 0.5

    def test_undefined_without_relevant_documents(self):
        import pytest

        from eval.retrieval import r_precision

        with pytest.raises(ValueError, match="undefined"):
            r_precision(["a"], set())


class TestAveragePrecision:
    """MAP is the metric ragas reports as `context_precision`, which is a
    different thing from precision_at_k despite the shared word - so these
    pin the definition down by value, not by shape."""

    def test_perfect_ranking_scores_one(self):
        from eval.retrieval import average_precision_at_k

        assert average_precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0

    def test_rewards_hits_that_rank_higher(self):
        """The whole reason to report it beside recall: recall cannot tell
        these two apart, and the generator reads top-down."""
        from eval.retrieval import average_precision_at_k

        early = average_precision_at_k(["a", "x", "x"], {"a"}, k=3)
        late = average_precision_at_k(["x", "x", "a"], {"a"}, k=3)

        assert early > late

    def test_precision_is_averaged_over_the_hits_not_the_slots(self):
        """Relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333. Divided by the
        number of relevant documents, not by k - dividing by k would make a
        perfect ranking score 2/3 here."""
        from eval.retrieval import average_precision_at_k

        assert average_precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(
            (1.0 + 2 / 3) / 2
        )

    def test_capped_by_k_when_relevant_outnumbers_the_slots(self):
        """Three gold chunks and two slots: a perfect top-2 is the best
        possible, so it must score 1.0 rather than 2/3."""
        from eval.retrieval import average_precision_at_k

        assert average_precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0

    def test_deduped_like_the_other_metrics(self):
        from eval.retrieval import average_precision_at_k

        assert average_precision_at_k(["a", "a"], {"a"}, k=2) == 1.0

    def test_nothing_relevant_retrieved_scores_zero(self):
        from eval.retrieval import average_precision_at_k

        assert average_precision_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_undefined_without_relevant_documents(self):
        from eval.retrieval import average_precision_at_k

        with pytest.raises(ValueError, match="undefined"):
            average_precision_at_k(["a"], set(), k=2)
