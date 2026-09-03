"""The free branch: retrieval scoring and the ablation grids.

No LLM, no Pinecone, no Postgres - the search engine is stubbed so what is
under test is the scoring and the grid shapes, not the I/O.
"""
from unittest.mock import MagicMock

from eval.ablations import grid_a, grid_b, pareto
from eval.run_free import by_topic, confidence_interval, curated


class TestCurated:
    def test_scores_only_kept_questions(self):
        rows = [{"decision": "keep", "id": "a"}, {"decision": "drop", "id": "b"}]

        assert [q["id"] for q in curated(rows)] == ["a"]

    def test_falls_back_to_everything_before_review_starts(self):
        """The branch stays usable during the review pass - the numbers move
        as questions are kept, which is the point of it being free."""
        rows = [{"decision": "undecided", "id": "a"}, {"decision": "undecided", "id": "b"}]

        assert len(curated(rows)) == 2


class TestConfidenceInterval:
    def test_interval_shrinks_with_more_questions(self):
        """The interval comes from spread across questions, not repeated
        runs - which is why it costs nothing to report."""
        few = confidence_interval([0.0, 1.0] * 5)
        many = confidence_interval([0.0, 1.0] * 50)

        assert many["ci"] < few["ci"]

    def test_identical_scores_have_no_interval(self):
        assert confidence_interval([0.5] * 10)["ci"] == 0.0

    def test_single_question_reports_no_interval_rather_than_dividing_by_zero(self):
        assert confidence_interval([0.7]) == {"mean": 0.7, "ci": 0.0, "n": 1}

    def test_empty_is_zero_not_an_error(self):
        assert confidence_interval([])["n"] == 0


class TestByTopic:
    def test_a_question_counts_toward_every_topic_it_spans(self):
        """Multi-hop questions cross topics; dropping them from one side
        would understate whichever topic they were assigned away from."""
        per_q = [
            {"abstention": False, "ndcg": 1.0, "topics": ["agents", "retrieval"]},
            {"abstention": False, "ndcg": 0.0, "topics": ["agents"]},
        ]
        result = by_topic(per_q)

        assert result["agents"]["n"] == 2 and result["retrieval"]["n"] == 1

    def test_abstention_questions_are_excluded(self):
        per_q = [{"abstention": True, "topics": ["agents"]}]

        assert by_topic(per_q) == {}


class TestGridA:
    def test_is_modes_times_top_ks_with_reranking_off(self):
        grid = grid_a()

        assert len(grid) == 18 and not any(c["rerank"] for c in grid)


class TestGridB:
    def test_pins_top_k_to_the_pool(self):
        """search() computes retrieve_k = max(candidates, top_k), so sweeping
        both produces duplicate configs: (top_k=50, candidates=20) and
        (top_k=50, candidates=50) are the same retrieval."""
        assert all(c["top_k"] == c["candidates"] for c in grid_b("hybrid"))

    def test_never_returns_more_than_the_pool_holds(self):
        assert all(c["rerank_top_k"] <= c["candidates"] for c in grid_b("hybrid"))

    def test_sweeps_the_score_floor_inside_the_grid(self):
        """The floor is the abstention mechanism. Measuring it in a separate
        script is why every historical sweep reported abstention_precision
        0.000 for every row."""
        floors = {c["min_rerank_score"] for c in grid_b("hybrid")}

        assert None in floors and len(floors) > 1

    def test_pool_ladder_stops_at_forty(self):
        """Reranking cannot recover a chunk the first stage never retrieved,
        so the ceiling comes from ablation A. Widen only if recall@40 is
        still climbing there."""
        assert max(c["candidates"] for c in grid_b("hybrid")) == 40


class TestPareto:
    def test_keeps_configs_nothing_dominates_on_both_axes(self):
        results = [
            {"recall": 0.9, "precision": 0.1},
            {"recall": 0.5, "precision": 0.5},
            {"recall": 0.4, "precision": 0.4},  # dominated by the row above
        ]
        frontier = pareto(results)

        assert {"recall": 0.4, "precision": 0.4} not in frontier
        assert len(frontier) == 2

    def test_selecting_on_one_metric_alone_picks_its_extreme(self):
        """Which is why the frontier exists: argmax-recall is always the
        widest k, and says nothing about what to ship."""
        results = [{"recall": 0.9, "precision": 0.1}, {"recall": 0.5, "precision": 0.5}]

        assert max(results, key=lambda r: r["recall"])["precision"] == 0.1
        assert len(pareto(results)) == 2


class TestRunConfigScoring:
    def test_judges_at_the_number_actually_returned_when_reranking(self):
        """With reranking on, rerank_top_k is the output size - scoring at
        the retrieval depth would divide precision by a k the caller never
        saw."""
        from eval.ablations import run_config

        engine = MagicMock()
        engine.search.return_value = MagicMock(
            results=[MagicMock(chunk_id=1), MagicMock(chunk_id=2)]
        )
        rows = [{"id": "q", "question": "?", "reference_chunk_ids": [1]}]
        cfg = {
            "mode": "hybrid",
            "top_k": 40,
            "rerank": True,
            "rerank_top_k": 2,
            "candidates": 40,
        }

        # precision = 1 relevant / k=2 returned, not / k=40 retrieved
        assert run_config(engine, rows, cfg)["precision"] == 0.5

    def test_expansion_is_off_so_the_scored_unit_matches_the_seeded_unit(self):
        from eval.ablations import run_config

        engine = MagicMock()
        engine.search.return_value = MagicMock(results=[])
        rows = [{"id": "q", "question": "?", "reference_chunk_ids": [1]}]
        cfg = {"mode": "hybrid", "top_k": 5, "rerank": False, "rerank_top_k": 5}

        run_config(engine, rows, cfg)

        assert engine.search.call_args.kwargs["neighbour_window"] == 0
