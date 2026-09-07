"""Cost controls on the judged branch: stratified sampling and token accounting.

Same isolation rules as tests/test_eval_run.py, for the same reason: importing
eval.run pulls in ragas, whose executor calls nest_asyncio.apply() at import
time and poisons the process-wide asyncio event loop for the anyio-based
FastAPI TestClient tests. Marked `eval` AND kept out of default collection via
pyproject's --ignore (a marker alone isn't enough - pytest imports every
collected file just to read its markers). Run with:

    uv run pytest -o addopts="" tests/test_eval_sampling.py -m eval
"""
import pytest

pytestmark = pytest.mark.eval

SINGLE = "single_hop_specifc_query_synthesizer"
ABSTRACT = "multi_hop_abstract_query_synthesizer"
SPECIFIC = "multi_hop_specific_query_synthesizer"


def _rows(spec: list[tuple[str, int]]) -> list[dict]:
    """spec: (synthesizer, how_many) -> flat rows in the curated set's shape.

    Ids are numbered across the whole set, not per synthesizer: the two
    multi-hop names share a prefix, and a per-bucket scheme collides.
    """
    out = []
    for synth, count in spec:
        for i in range(count):
            out.append({
                "id": f"q{len(out):04d}",
                "question": f"{synth}/{i}",
                "reference": "gold",
                "reference_chunk_ids": [i],
                "synthesizer": synth,
            })
    return out


# The curated set's real shape: 46 single-hop, 33 abstract, 21 specific.
CURATED = _rows([(SINGLE, 46), (ABSTRACT, 33), (SPECIFIC, 21)])


class TestStratifiedSample:
    """Stratified on synthesizer because that is what the scores actually
    split on - single-hop recall runs near 0.95 against 0.5-0.6 for multi-hop,
    so an unstratified sample of 10 reports whichever type it happened to
    draw."""

    def test_returns_exactly_n(self):
        from eval.run import stratified_sample

        assert len(stratified_sample(CURATED, 20)) == 20

    def test_every_stratum_is_represented_when_n_allows(self):
        from eval.run import stratified_sample

        got = stratified_sample(CURATED, 20)

        assert {r["synthesizer"] for r in got} == {SINGLE, ABSTRACT, SPECIFIC}

    def test_shares_track_the_population(self):
        """46/33/21 in, roughly 46/33/21 out - a sample that over-weights
        single-hop would report a flattering number."""
        from eval.run import stratified_sample

        got = stratified_sample(CURATED, 50)
        n_single = sum(1 for r in got if r["synthesizer"] == SINGLE)

        assert 18 <= n_single <= 28

    def test_small_strata_survive(self):
        from eval.run import stratified_sample

        rows = _rows([(SINGLE, 40), (SPECIFIC, 2)])

        assert len(stratified_sample(rows, 21)) == 21

    def test_is_deterministic_across_calls(self):
        from eval.run import stratified_sample

        assert stratified_sample(CURATED, 15) == stratified_sample(CURATED, 15)

    def test_different_seeds_pick_differently(self):
        from eval.run import stratified_sample

        a = stratified_sample(CURATED, 15, seed=0)
        b = stratified_sample(CURATED, 15, seed=7)

        assert [r["id"] for r in a] != [r["id"] for r in b]

    def test_preserves_dataset_order(self):
        """A sample that reorders the set makes two runs harder to diff line
        by line, for no benefit."""
        from eval.run import stratified_sample

        got = stratified_sample(CURATED, 15)
        ids = [r["id"] for r in CURATED]

        assert [ids.index(r["id"]) for r in got] == sorted(ids.index(r["id"]) for r in got)

    def test_rows_are_the_original_objects_not_copies(self):
        from eval.run import stratified_sample

        assert all(any(r is o for o in CURATED) for r in stratified_sample(CURATED, 10))

    def test_a_sample_no_smaller_than_the_set_returns_everything(self):
        from eval.run import stratified_sample

        assert stratified_sample(CURATED, 500) is CURATED


class TestMetricSelection:
    def test_runs_the_four_a_judge_is_needed_for(self):
        """Context precision and recall were dropped when eval/sweep.py
        measured ranking free against *document* labels. The labels are
        chunk-level now and the free branch measures them exactly, so these
        two are no longer a weaker copy of something already measured - they
        are the judge's different answer to the same question."""
        from eval.run import METRICS

        assert {m.name for m in METRICS} == {
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
            "context_recall",
        }


class TestJudgeSpend:
    def test_reports_dollars_for_a_priced_model(self):
        from ragas.cost import TokenUsage

        from eval.run import judge_spend

        result = type("R", (), {"total_tokens": lambda self: TokenUsage(
            input_tokens=1_000_000, output_tokens=1_000_000)})()

        assert judge_spend(result, "deepseek/deepseek-v4-flash")["usd"] > 0

    def test_records_tokens_without_inventing_a_price(self):
        """An unlisted model must report usage and omit cost rather than make
        a number up."""
        from ragas.cost import TokenUsage

        from eval.run import judge_spend

        result = type("R", (), {"total_tokens": lambda self: TokenUsage(
            input_tokens=10, output_tokens=5)})()
        spend = judge_spend(result, "some/unlisted-model")

        assert spend["input_tokens"] == 10 and "usd" not in spend

    def test_a_run_that_recorded_nothing_does_not_raise(self):
        """Accounting must never fail a run whose calls are already paid for."""
        from eval.run import judge_spend

        def boom(self):
            raise RuntimeError("no usage recorded")

        assert judge_spend(type("R", (), {"total_tokens": boom})(), "x") == {}
