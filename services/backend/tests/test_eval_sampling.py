"""Tests for the cost controls added in #26: stratified sampling and judge
token accounting.

Same isolation rules as tests/test_eval_run.py, for the same reason:
importing eval.run pulls in ragas, whose executor calls nest_asyncio.apply()
at import time and poisons the process-wide asyncio event loop for the
anyio-based FastAPI TestClient tests. Marked `eval` AND kept out of default
collection via pyproject's --ignore (a marker alone isn't enough - pytest
imports every collected file just to read its markers). Run with:

    uv run pytest -o addopts="" tests/test_eval_sampling.py -m eval

The helpers under test are pure functions over dataset rows and a usage
object, so none of this costs an API call.
"""
import pytest

pytestmark = pytest.mark.eval


def _rows(spec: list[tuple[str, str, int]]) -> list[dict]:
    """spec: (category, domain, how_many) -> flat dataset rows."""
    out = []
    for category, domain, count in spec:
        for i in range(count):
            out.append(
                {
                    "category": category,
                    "domain": domain,
                    "question": f"{category}/{domain}/{i}",
                    "ground_truth": "gt",
                }
            )
    return out


class TestStratifiedSample:
    def test_returns_everything_when_n_covers_the_dataset(self):
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 3)])

        assert stratified_sample(rows, 3) == rows
        assert stratified_sample(rows, 99) == rows

    def test_returns_exactly_n(self):
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 20), ("grounded", "ml", 20)])

        assert len(stratified_sample(rows, 7)) == 7

    def test_small_strata_survive(self):
        """The reason this isn't a uniform random sample. edge_case is 2 rows
        out of 42; proportional allotment rounds it to zero, and abstention
        regressions then go unmeasured while the score still looks complete."""
        from eval.run import stratified_sample

        rows = _rows(
            [
                ("grounded", "robotics", 20),
                ("grounded", "ml", 20),
                ("edge_case", "", 2),
            ]
        )

        picked = stratified_sample(rows, 9)

        assert sum(1 for r in picked if r["category"] == "edge_case") >= 1

    def test_every_stratum_represented_when_n_allows(self):
        from eval.run import stratified_sample

        rows = _rows(
            [
                ("grounded", "robotics", 10),
                ("grounded", "ml", 10),
                ("grounded", "bio", 10),
                ("edge_case", "", 4),
            ]
        )

        picked = stratified_sample(rows, 8)
        strata = {(r["category"], r["domain"]) for r in picked}

        assert len(strata) == 4

    def test_is_deterministic_across_calls(self):
        """A subset score is only comparable week to week if `--sample 15`
        keeps meaning the same 15 questions."""
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 20), ("edge_case", "", 5)])

        first = stratified_sample(rows, 11)
        second = stratified_sample(rows, 11)

        assert [r["question"] for r in first] == [r["question"] for r in second]

    def test_different_seeds_pick_differently(self):
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 30)])

        a = [r["question"] for r in stratified_sample(rows, 10, seed=0)]
        b = [r["question"] for r in stratified_sample(rows, 10, seed=99)]

        assert a != b

    def test_preserves_dataset_order(self):
        """The report's per-question table should read in the same sequence as
        the file, not in whatever order the shuffle happened to produce."""
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 10), ("grounded", "ml", 10)])
        position = {r["question"]: i for i, r in enumerate(rows)}

        picked = stratified_sample(rows, 8)
        got = [position[r["question"]] for r in picked]

        assert got == sorted(got)

    def test_rows_are_the_original_objects_not_copies(self):
        from eval.run import stratified_sample

        rows = _rows([("grounded", "robotics", 6)])

        picked = stratified_sample(rows, 3)

        assert all(any(p is r for r in rows) for p in picked)


class TestJudgeSpend:
    def test_reports_tokens_and_cost_for_a_known_model(self):
        from eval.run import judge_spend

        class Result:
            def total_tokens(self):
                from ragas.cost import TokenUsage

                return TokenUsage(input_tokens=1_000_000, output_tokens=200_000)

        spend = judge_spend(Result(), "claude-haiku-4-5")

        assert spend["input_tokens"] == 1_000_000
        assert spend["output_tokens"] == 200_000
        # $1/MTok in + $5/MTok out = 1.00 + 1.00
        assert spend["usd"] == pytest.approx(2.00)

    def test_sums_usage_when_ragas_reports_per_model(self):
        from eval.run import judge_spend

        class Result:
            def total_tokens(self):
                from ragas.cost import TokenUsage

                return [
                    TokenUsage(input_tokens=10, output_tokens=1),
                    TokenUsage(input_tokens=5, output_tokens=2),
                ]

        spend = judge_spend(Result(), "claude-haiku-4-5")

        assert spend["input_tokens"] == 15
        assert spend["output_tokens"] == 3

    def test_records_tokens_but_omits_cost_for_an_unpriced_model(self):
        from eval.run import judge_spend

        class Result:
            def total_tokens(self):
                from ragas.cost import TokenUsage

                return TokenUsage(input_tokens=42, output_tokens=7)

        spend = judge_spend(Result(), "some-self-hosted-model")

        assert spend == {"input_tokens": 42, "output_tokens": 7}

    def test_missing_usage_is_not_fatal(self):
        """Ragas raises when evaluate() ran without a token_usage_parser. By
        the time this is called every judge call is already paid for, so
        bookkeeping must not be what destroys the run."""
        from eval.run import judge_spend

        class Result:
            def total_tokens(self):
                raise ValueError("not configured for computing cost")

        assert judge_spend(Result(), "claude-haiku-4-5") == {}


class TestMetricSelection:
    def test_only_the_two_metrics_a_judge_is_needed_for(self):
        """#26: context_precision/context_recall are measured judge-free and
        against labels by eval/sweep.py. Re-adding them here silently
        multiplies the bill - context_precision issues one call per retrieved
        context, so its cost scales with k."""
        from eval.run import METRICS

        assert {m.name for m in METRICS} == {"faithfulness", "answer_relevancy"}

    def test_answer_relevancy_generates_one_question_not_three(self):
        from eval.run import METRICS

        relevancy = next(m for m in METRICS if m.name == "answer_relevancy")

        assert relevancy.strictness == 1


class TestTokenUsageParser:
    """#18 — ragas reads token counts out of the raw provider response, and
    the shapes differ: Anthropic reports `usage.input_tokens`, OpenAI-style
    endpoints report `token_usage.prompt_tokens`. The wrong parser doesn't
    fail, it silently returns zeros — which would gut the cost reporting the
    harness exists to provide."""

    def _cfg(self, provider):
        from config import Config, LLMConfig

        return Config(llm=LLMConfig(provider=provider))

    def test_anthropic_provider_uses_the_anthropic_parser(self):
        from ragas.cost import get_token_usage_for_anthropic

        from eval.run import token_usage_parser

        assert token_usage_parser(self._cfg("anthropic")) is get_token_usage_for_anthropic

    def test_openrouter_uses_the_openai_parser(self):
        from ragas.cost import get_token_usage_for_openai

        from eval.run import token_usage_parser

        assert token_usage_parser(self._cfg("openrouter")) is get_token_usage_for_openai

    def test_a_free_model_prices_at_zero_not_unknown(self):
        """':free' really is $0 — the tokens still count and are still worth
        reporting, they just cost nothing. Reporting 'unknown' would hide a
        number we actually know."""
        from eval.run import judge_price

        assert judge_price("minimax/minimax-m2.7:free") == (0.0, 0.0)

    def test_claude_through_openrouter_is_priced(self):
        from eval.run import judge_price

        assert judge_price("anthropic/claude-haiku-4.5") == (1.00 / 1e6, 5.00 / 1e6)

    def test_an_unknown_paid_model_has_no_price(self):
        from eval.run import judge_price

        assert judge_price("some/unlisted-model") is None
