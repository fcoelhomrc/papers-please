"""LLM query-side arms - the parsing, caching and fusion, with no LLM involved.

These arms live in the free branch on one property: a transformed query is a
pure function of the question, so the LLM runs once and every later sweep
re-reads the cache. The tests that matter are therefore the ones guarding that
cache's correctness and the fallbacks that stop a bad transform from being
scored as a retrieval failure.
"""
from types import SimpleNamespace

import pytest

from eval.query_arms import (
    ARMS,
    DECOMPOSE,
    DENSE_ONLY,
    HYDE,
    MULTI_QUERY,
    cache_key,
    cache_path,
    load_cached,
    parse,
    parse_lines,
    retrieve_for,
    save_cached,
)


class TestParse:
    def test_strips_numbering_the_model_adds_anyway(self):
        """The prompt says no numbering; models add it regardless, and a query
        beginning "1. " searches for the digit."""
        assert parse_lines("1. how does RAG work\n2) what is HyDE", 3) == [
            "how does RAG work",
            "what is HyDE",
        ]

    def test_strips_bullets(self):
        assert parse_lines("- alpha\n* beta", 3) == ["alpha", "beta"]

    def test_drops_blank_lines(self):
        assert parse_lines("alpha\n\n\nbeta", 3) == ["alpha", "beta"]

    def test_caps_at_the_limit(self):
        """A model that returns six paraphrases would otherwise cost six
        retrievals per question and silently change what the arm is."""
        assert len(parse_lines("a\nb\nc\nd\ne\nf", 3)) == 3

    def test_hyde_keeps_its_passage_whole(self):
        """A HyDE passage is one embedding target. Splitting it on newlines
        would embed each paragraph separately and search for the first."""
        passage = "Quantization reduces precision.\n\nWe evaluate on WinoGrande."

        assert parse(HYDE, passage) == [passage]

    def test_other_arms_split_on_lines(self):
        assert len(parse(MULTI_QUERY, "one\ntwo\nthree")) == 3


class TestCacheKey:
    """Both the model and the prompt version change the queries, so a change in
    either has to invalidate - otherwise a prompt edit reports the new prompt's
    score for the old prompt's output."""

    def test_model_change_invalidates(self):
        assert cache_key("a", "v1") != cache_key("b", "v1")

    def test_prompt_version_change_invalidates(self):
        assert cache_key("a", "v1") != cache_key("a", "v2")


class TestCacheRoundTrip:
    def test_reads_back_what_it_wrote(self, tmp_path, monkeypatch):
        import eval.query_arms as qa

        monkeypatch.setattr(qa, "TESTSET_DIR", tmp_path)
        save_cached(MULTI_QUERY, {"q0001": ["a", "b"]}, "m", "v1")

        assert load_cached(MULTI_QUERY, "m", "v1") == {"q0001": ["a", "b"]}

    def test_a_stale_key_reads_as_a_miss_not_as_data(self, tmp_path, monkeypatch):
        import eval.query_arms as qa

        monkeypatch.setattr(qa, "TESTSET_DIR", tmp_path)
        save_cached(MULTI_QUERY, {"q0001": ["a"]}, "m", "v1")

        assert load_cached(MULTI_QUERY, "m", "v2") is None

    def test_a_missing_file_reads_as_a_miss(self, tmp_path, monkeypatch):
        import eval.query_arms as qa

        monkeypatch.setattr(qa, "TESTSET_DIR", tmp_path)

        assert load_cached(DECOMPOSE, "m", "v1") is None

    def test_each_arm_has_its_own_file(self, tmp_path, monkeypatch):
        import eval.query_arms as qa

        monkeypatch.setattr(qa, "TESTSET_DIR", tmp_path)

        assert len({cache_path(a) for a in ARMS}) == len(ARMS)


def _engine(calls):
    """A stub recording which retrieval path each arm reaches for."""

    def record(name):
        def fn(query, top_k, *a, **kw):
            calls.append((name, query))
            return [{"chunk_id": hash(query) % 1000, "score": 1.0}]

        return fn

    return SimpleNamespace(
        _vector_candidates=record("vector"),
        _bm25_candidates=record("bm25"),
        _keyword_candidates=record("keyword"),
    )


CFG = SimpleNamespace(rrf_k=60, hybrid_candidates=20, keyword_weight=0.1)


class TestRetrieveFor:
    def test_hyde_always_goes_dense_whatever_the_mode(self):
        """HyDE's premise is that the *embedding* of a paper-shaped passage
        sits near the real passage. Handing that passage to a lexical retriever
        searches for words the model invented - a different, worse technique."""
        calls = []

        retrieve_for(_engine(calls), HYDE, ["a passage"], 10, CFG, mode="bm25")

        assert [name for name, _ in calls] == ["vector"]

    def test_hyde_is_declared_dense_only(self):
        assert HYDE in DENSE_ONLY and MULTI_QUERY not in DENSE_ONLY

    def test_multi_query_retrieves_once_per_query(self):
        calls = []

        retrieve_for(_engine(calls), MULTI_QUERY, ["a", "b", "c"], 10, CFG, mode="bm25")

        assert [q for _, q in calls] == ["a", "b", "c"]

    def test_a_single_query_skips_fusion(self):
        """Decomposition returns one line when a question has one lookup;
        fusing a single list is a no-op that would still reorder by RRF score."""
        calls = []

        out = retrieve_for(_engine(calls), DECOMPOSE, ["only"], 10, CFG, mode="bm25")

        assert len(calls) == 1 and out[0]["score"] == 1.0

    def test_each_sub_query_is_pulled_at_full_top_k(self):
        """Not top_k/n: a chunk ranked 8th by every sub-query would be cut
        before fusion could notice the agreement."""
        calls = []
        engine = SimpleNamespace(
            _bm25_candidates=lambda q, k, *a, **kw: calls.append(k) or [],
            _vector_candidates=lambda q, k, *a, **kw: [],
            _keyword_candidates=lambda q, k, *a, **kw: [],
        )

        retrieve_for(engine, MULTI_QUERY, ["a", "b"], 10, CFG, mode="bm25")

        assert calls == [10, 10]

    def test_a_hybrid_mode_is_refused_rather_than_silently_wrong(self):
        """The arms fuse across sub-queries; nesting that inside a mode that
        already fuses across retrievers would apply RRF twice with undeclared
        weights."""
        with pytest.raises(ValueError, match="single-source"):
            retrieve_for(_engine([]), MULTI_QUERY, ["a"], 10, CFG, mode="hybrid")


# --- which modes an arm may run in ---------------------------------------
#
# One definition, because it used to be two: `DENSE_ONLY` here and
# `ablations.ARM_MODES` there, with the judged branch checking neither - which
# is how `--mode bm25 --arm hyde` became a dense run filed under results/bm25/.


def test_hyde_is_semantic_only():
    from eval.query_arms import modes_for, supports

    assert modes_for("hyde") == ("semantic",)
    assert supports("hyde", "semantic")
    assert not supports("hyde", "bm25")
    assert not supports("hyde", "keyword")


def test_ordinary_arms_run_in_any_single_source_mode():
    from eval.query_arms import supports

    for arm in ("none", "multi_query", "decompose"):
        for mode in ("semantic", "bm25", "keyword"):
            assert supports(arm, mode), (arm, mode)


def test_hybrid_is_never_supported():
    """retrieve_for fuses sub-queries; fusing a second retriever too is a
    different experiment, and _single would raise on it anyway."""
    from eval.query_arms import supports

    for arm in ("none", "multi_query", "hyde"):
        assert not supports(arm, "hybrid")


def test_orig_variants_inherit_the_base_arms_modes():
    from eval.query_arms import modes_for

    assert modes_for("multi_query+orig") == modes_for("multi_query")
    assert modes_for("hyde+orig") == ("semantic",)


def test_ablations_table_derives_from_supports():
    """The sweep's table must not drift from the rule it encodes."""
    from eval.ablations import ARM_MODES, ARMS, SWEPT_MODES
    from eval.query_arms import supports

    for arm in ARMS:
        assert ARM_MODES[arm] == tuple(m for m in SWEPT_MODES if supports(arm, m))
    assert ARM_MODES["hyde"] == ("semantic",)
