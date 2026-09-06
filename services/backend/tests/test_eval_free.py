"""The free retrieval branch: exact chunk-id scoring and its intervals.

No LLM anywhere, so this runs on every change - which only holds if the two
correctness rules hold. Scoring must use the retrieved chunk, not the
neighbour-expanded context, and confidence intervals must come from the
spread across questions rather than from repeated calls.
"""
import math

from eval.retrieval import RETRIEVAL_METRICS
from eval.run_free import by_group, mean_ci, paired_diff_ci, retrieve, summarise


class TestMeanCi:
    def test_interval_shrinks_as_questions_are_added(self):
        """The variance that matters is across questions, so more questions
        is the only thing that tightens it - no extra calls involved."""
        small = mean_ci([0.0, 1.0] * 5)[1]
        large = mean_ci([0.0, 1.0] * 50)[1]

        assert large < small

    def test_zero_spread_gives_a_zero_interval(self):
        assert mean_ci([0.5] * 20) == (0.5, 0.0)

    def test_single_question_reports_no_interval_rather_than_inventing_one(self):
        """One observation has no spread to estimate from."""
        assert mean_ci([0.7]) == (0.7, 0.0)

    def test_empty_is_not_an_error(self):
        assert mean_ci([]) == (0.0, 0.0)

    def test_matches_the_textbook_formula(self):
        values = [0.0, 0.5, 1.0]
        mean, half = mean_ci(values)
        sd = math.sqrt(sum((v - 0.5) ** 2 for v in values) / 2)

        assert (round(mean, 6), round(half, 6)) == (0.5, round(1.96 * sd / math.sqrt(3), 6))


class TestPairedDiff:
    def test_paired_interval_is_tighter_than_comparing_two_means(self):
        """Both configurations answer the same questions, so difficulty
        cancels. Two overlapping individual CIs do not imply the difference
        is indistinguishable from zero - which is why A-vs-B uses this."""
        a = [0.1, 0.5, 0.9, 0.3, 0.7]
        b = [0.0, 0.4, 0.8, 0.2, 0.6]

        assert paired_diff_ci(a, b)[1] < mean_ci(a)[1]

    def test_reports_the_mean_difference(self):
        assert round(paired_diff_ci([0.5, 0.7], [0.3, 0.5])[0], 6) == 0.2


class TestRetrieveScoresTheRetrievedChunk:
    def test_neighbour_expansion_is_disabled(self):
        """`context` glues neighbouring chunks onto a hit. Scoring against it
        would credit retrieval for chunks it never returned."""
        from unittest.mock import MagicMock

        engine = MagicMock()
        engine.search.return_value = MagicMock(results=[])

        retrieve(engine, "q", {"mode": "hybrid", "top_k": 5, "rerank": False})

        assert engine.search.call_args.kwargs["neighbour_window"] == 0

    def test_returns_chunk_ids_with_their_text_in_rank_order(self):
        """The text rides along because the ragas non-LLM metrics score
        strings while the chunk-id metrics score integers, and both are
        computed from this one retrieval."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        engine = MagicMock()
        engine.search.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(chunk_id=9, text="nine"),
                SimpleNamespace(chunk_id=4, text="four"),
            ]
        )

        assert retrieve(engine, "q", {"mode": "hybrid", "top_k": 5, "rerank": False}) == [
            (9, "nine"),
            (4, "four"),
        ]


def scored(**overrides):
    """A per-question score row carrying every metric summarise() reports.

    Built from RETRIEVAL_METRICS rather than listed by hand: these fixtures
    broke the whole module the last time a metric was added, which is a test
    failing for a reason unrelated to what it tests."""
    row = {"abstention": False, **{m: 0.5 for m in RETRIEVAL_METRICS}}
    return {**row, **overrides}


class TestSummarise:
    def _q(self, recall, **extra):
        return scored(recall=recall, precision=0.1, hit_rate=1.0, **extra)

    def test_every_metric_carries_an_interval(self):
        out = summarise([self._q(0.4), self._q(0.8)])

        assert {f"{m}_ci" for m in RETRIEVAL_METRICS} <= set(out)

    def test_reports_the_number_of_answerable_questions(self):
        assert summarise([self._q(1.0), self._q(0.0)])["n"] == 2


class TestByGroup:
    def test_splits_a_multi_valued_key_across_every_value(self):
        """A multi-hop question spans more than one topic and belongs to
        each of its slices."""
        q = scored(topics=["agents", "retrieval"])

        assert set(by_group([q], "topics")) == {"agents", "retrieval"}

    def test_handles_a_single_valued_key(self):
        q = scored(synthesizer="single_hop_specific_query_synthesizer")

        assert list(by_group([q], "synthesizer")) == [
            "single_hop_specific_query_synthesizer"
        ]
