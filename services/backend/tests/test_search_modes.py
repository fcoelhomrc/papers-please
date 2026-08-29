"""Unit tests for retrieval modes and RRF fusion.

No Pinecone, no Postgres, no models: rrf_fuse is a pure function, and the
mode dispatch is tested by stubbing the two candidate sources so what's
under test is the routing/fusion, not the I/O.
"""
from unittest.mock import MagicMock, patch

import pytest

from search import HYBRID, KEYWORD, RETRIEVAL_MODES, SEMANTIC, SearchEngine, rrf_fuse


def chunk(cid, score=0.0, text=None):
    return {
        "chunk_id": cid,
        "doc_id": cid * 10,
        "text": text or f"chunk {cid}",
        "page_num": 1,
        "pdf_path": f"{cid}.pdf",
        "title": f"Paper {cid}",
        "authors": ["A. Author"],
        "year": 2024,
        "score": score,
    }


class TestRrfFuse:
    def test_agreement_between_sources_outranks_a_single_top_hit(self):
        """The whole point of fusion: a chunk both retrievers like beats one
        that only the top of a single list liked."""
        vector = [chunk(1), chunk(2)]
        keyword = [chunk(3), chunk(2)]

        fused = rrf_fuse([vector, keyword], k=60)

        # 2 appears at rank 2 in both: 1/62 + 1/62 > 1/61 (rank 1, one list)
        assert [c["chunk_id"] for c in fused][0] == 2

    def test_score_is_the_rrf_score_not_the_source_score(self):
        fused = rrf_fuse([[chunk(1, score=0.99)]], k=60)
        assert fused[0]["score"] == pytest.approx(1 / 61)

    def test_ranks_are_one_based(self):
        """Off-by-one here silently changes every fused ranking."""
        fused = rrf_fuse([[chunk(1)]], k=0)
        assert fused[0]["score"] == pytest.approx(1.0)  # 1/(0+1), not 1/(0+0)

    def test_union_not_intersection(self):
        fused = rrf_fuse([[chunk(1)], [chunk(2)]], k=60)
        assert sorted(c["chunk_id"] for c in fused) == [1, 2]

    def test_preserves_chunk_payload(self):
        fused = rrf_fuse([[chunk(7, text="the passage")]], k=60)
        assert fused[0]["text"] == "the passage"
        assert fused[0]["title"] == "Paper 7"

    def test_empty_sources_fuse_to_empty(self):
        assert rrf_fuse([[], []], k=60) == []

    def test_one_empty_source_still_returns_the_other(self):
        """Keyword search returns nothing for a query with no literal matches
        - hybrid must degrade to the other source, not to nothing."""
        fused = rrf_fuse([[chunk(1), chunk(2)], []], k=60)
        assert [c["chunk_id"] for c in fused] == [1, 2]

    def test_smaller_k_sharpens_top_rank_dominance(self):
        vector = [chunk(1), chunk(2)]
        keyword = [chunk(3), chunk(2)]
        # With k=0 the rank-1 entries (1/1) beat the doubled rank-2 (1/2+1/2)
        # only on ties; check the ordering actually responds to k at all.
        assert rrf_fuse([vector, keyword], k=0)[0]["chunk_id"] != 2
        assert rrf_fuse([vector, keyword], k=60)[0]["chunk_id"] == 2


def make_engine():
    engine = SearchEngine.__new__(SearchEngine)  # skip __init__ (Pinecone/DB)
    engine._model_key = "bge-small"
    engine._reranker = MagicMock()
    return engine


class TestSearchModeDispatch:
    def test_semantic_uses_only_the_vector_source(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])

        resp = engine.search("q", top_k=5, mode=SEMANTIC)

        engine._vector_candidates.assert_called_once_with("q", 5)
        engine._keyword_candidates.assert_not_called()
        assert resp.mode == "semantic"
        assert [r.chunk_id for r in resp.results] == [1]

    def test_keyword_uses_only_the_keyword_source(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])

        resp = engine.search("q", top_k=5, mode=KEYWORD)

        engine._keyword_candidates.assert_called_once_with("q", 5)
        engine._vector_candidates.assert_not_called()
        assert [r.chunk_id for r in resp.results] == [2]

    def test_hybrid_queries_both_and_fuses(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1), chunk(2)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(3), chunk(2)])

        resp = engine.search("q", top_k=5, mode=HYBRID)

        assert engine._vector_candidates.called and engine._keyword_candidates.called
        assert resp.mode == "hybrid"
        assert [r.chunk_id for r in resp.results][0] == 2  # agreed-on chunk wins

    def test_hybrid_pool_is_wider_than_top_k(self):
        """Each source must contribute more candidates than the final top_k,
        or fusion has nothing to reorder."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[])
        engine._keyword_candidates = MagicMock(return_value=[])

        engine.search("q", top_k=3, mode=HYBRID)

        pool = engine._vector_candidates.call_args.args[1]
        assert pool >= 20  # config default hybrid_candidates

    def test_hybrid_truncates_to_top_k_after_fusion(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(i) for i in range(1, 6)])
        engine._keyword_candidates = MagicMock(return_value=[])

        resp = engine.search("q", top_k=2, mode=HYBRID)

        assert len(resp.results) == 2

    def test_mode_defaults_to_config(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[])

        cfg = MagicMock()
        cfg.search.mode = "keyword"
        with patch("config.load", return_value=cfg):
            resp = engine.search("q", top_k=5)

        assert resp.mode == "keyword"
        engine._keyword_candidates.assert_called_once()

    def test_unknown_mode_raises(self):
        engine = make_engine()
        with pytest.raises(ValueError, match="unknown retrieval mode"):
            engine.search("q", mode="magic")

    def test_all_modes_are_reachable(self):
        assert set(RETRIEVAL_MODES) == {"semantic", "keyword", "hybrid"}


class TestRerankInteraction:
    def test_rerank_runs_after_fusion_on_the_fused_set(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1), chunk(2)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(3)])
        engine._reranker.rerank.return_value = [chunk(3, score=9.0)]

        resp = engine.search("q", top_k=5, rerank=True, rerank_top_k=1, mode=HYBRID)

        reranked_input = engine._reranker.rerank.call_args.args[1]
        assert {c["chunk_id"] for c in reranked_input} == {1, 2, 3}
        assert resp.reranked is True
        assert [r.chunk_id for r in resp.results] == [3]

    def test_no_rerank_call_when_nothing_retrieved(self):
        """Reranking an empty list is a wasted model call, and `reranked:
        true` on zero results misreports what happened."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[])

        resp = engine.search("q", rerank=True, mode=SEMANTIC)

        engine._reranker.rerank.assert_not_called()
        assert resp.reranked is False
        assert resp.results == []
