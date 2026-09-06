"""The ablation harness's fusion reconstruction - pure functions, no infra.

The sweep's whole economy rests on one property: candidates are cached
*unfused* per source, and every configuration is rebuilt from them in memory.
That is only sound if the rebuild reproduces what SearchEngine.search() would
have done - otherwise 60 configurations are cheap and wrong.
"""
from types import SimpleNamespace

from eval.ablations import LEXICAL_SIDE, candidates_for
from search import BM25, HYBRID, HYBRID_BM25, KEYWORD, SEMANTIC


def chunk(cid):
    return {"chunk_id": cid, "score": 1.0}


def sources(vector=(1, 2), keyword=(3, 4), bm25=(5, 6)):
    return {
        SEMANTIC: [chunk(c) for c in vector],
        KEYWORD: [chunk(c) for c in keyword],
        BM25: [chunk(c) for c in bm25],
    }


CFG = SimpleNamespace(hybrid_candidates=20, rrf_k=60, keyword_weight=0.1)


class TestSingleSourceModes:
    def test_each_mode_reads_its_own_cached_list(self):
        s = sources()

        picked = {
            mode: [c["chunk_id"] for c in candidates_for(s, mode, 2, CFG)]
            for mode in (SEMANTIC, KEYWORD, BM25)
        }

        assert picked == {SEMANTIC: [1, 2], KEYWORD: [3, 4], BM25: [5, 6]}


class TestHybridFusion:
    def test_the_two_hybrids_fuse_different_lexical_sides(self):
        """hybrid_bm25 exists so a reported number names the rankers behind
        it; if both hybrids read the same cached list the mode string lies."""
        s = sources()

        plain = {c["chunk_id"] for c in candidates_for(s, HYBRID, 4, CFG)}
        with_bm25 = {c["chunk_id"] for c in candidates_for(s, HYBRID_BM25, 4, CFG)}

        assert plain == {1, 2, 3, 4} and with_bm25 == {1, 2, 5, 6}

    def test_the_side_map_matches_the_engine(self):
        """Drift here would sweep a configuration production cannot produce."""
        from search import _HYBRID_KEYWORD_SIDE

        assert LEXICAL_SIDE == _HYBRID_KEYWORD_SIDE

    def test_defaults_reproduce_the_configured_fusion(self):
        """An unparameterised call has to be production exactly, or ablation W
        has no baseline to compare against."""
        s = sources()

        assert candidates_for(s, HYBRID, 4, CFG) == candidates_for(
            s, HYBRID, 4, CFG, keyword_weight=CFG.keyword_weight, rrf_k=CFG.rrf_k
        )

    def test_at_the_shipped_weight_a_keyword_hit_cannot_outrank_any_dense_hit(self):
        """The defect ablation W exists to expose. A rank-1 keyword hit scores
        0.1/61 = 0.00164; the *fifth* dense hit scores 1.0/65 = 0.01538. So the
        keyword side is outranked everywhere in the pool and contributes only
        chunks dense missed entirely - which is why hybrid measured the same as
        semantic."""
        s = sources(vector=(9, 8, 7, 6, 5), keyword=(3,))

        fused = [c["chunk_id"] for c in candidates_for(s, HYBRID, 6, CFG)]

        assert fused.index(3) == 5

    def test_at_parity_the_same_hit_ranks_by_its_own_rank(self):
        """Same inputs, weight 1.0: the keyword hit is rank 1 in its list, so
        it lands second overall behind the other rank-1 chunk."""
        s = sources(vector=(9, 8, 7, 6, 5), keyword=(3,))

        fused = [
            c["chunk_id"] for c in candidates_for(s, HYBRID, 6, CFG, keyword_weight=1.0)
        ]

        assert fused.index(3) == 1

    def test_rrf_k_changes_how_much_the_top_ranks_dominate(self):
        """Smaller k sharpens the reciprocal curve, so rank 1 pulls further
        ahead of rank 2. Swept because 60 is a paper default, not a
        measurement on this corpus."""
        s = sources(vector=(1, 2), keyword=(2, 1))

        sharp = candidates_for(s, HYBRID, 2, CFG, rrf_k=10)[0]["score"]
        damped = candidates_for(s, HYBRID, 2, CFG, rrf_k=60)[0]["score"]

        assert sharp > damped
