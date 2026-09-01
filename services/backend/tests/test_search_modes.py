"""Unit tests for retrieval modes and RRF fusion.

No Pinecone, no Postgres, no models: rrf_fuse is a pure function, and the
mode dispatch is tested by stubbing the two candidate sources so what's
under test is the routing/fusion, not the I/O.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from search import (
    HYBRID,
    KEYWORD,
    RETRIEVAL_MODES,
    SEMANTIC,
    SearchEngine,
    expand_neighbours,
    rrf_fuse,
)


def chunk(cid, score=0.0, text=None, chunk_index=None, obj_id=1):
    return {
        "chunk_id": cid,
        "chunk_index": cid if chunk_index is None else chunk_index,
        "obj_id": obj_id,
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
    # Neighbour expansion is on by default in config and needs a real
    # session; these tests are about mode dispatch and ranking, so it's
    # stubbed to a pass-through. expand_neighbours has its own tests below.
    engine._expand = MagicMock(side_effect=lambda chunks, window: chunks)
    return engine


class TestSearchModeDispatch:
    def test_semantic_uses_only_the_vector_source(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])

        resp = engine.search("q", top_k=5, mode=SEMANTIC)

        engine._vector_candidates.assert_called_once_with("q", 5, None)  # None = no floor
        engine._keyword_candidates.assert_not_called()
        assert resp.mode == "semantic"
        assert [r.chunk_id for r in resp.results] == [1]

    def test_keyword_uses_only_the_keyword_source(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])

        resp = engine.search("q", top_k=5, mode=KEYWORD)

        engine._keyword_candidates.assert_called_once_with("q", 5, None)
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


class TestThresholds:
    """Score floors let retrieval return nothing - the correct answer when the
    library has nothing relevant. Without them abstention_precision measured
    0.000: top_k came back regardless."""

    def test_vector_floor_drops_low_scoring_candidates(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[])
        engine._keyword_candidates = MagicMock(return_value=[])

        engine.search("q", top_k=5, mode=SEMANTIC, thresholds={"min_vector_score": 0.6})

        assert engine._vector_candidates.call_args.args[2] == 0.6

    def test_keyword_floor_passed_through(self):
        engine = make_engine()
        engine._keyword_candidates = MagicMock(return_value=[])

        engine.search("q", top_k=5, mode=KEYWORD, thresholds={"min_keyword_score": 0.05})

        assert engine._keyword_candidates.call_args.args[2] == 0.05

    def test_hybrid_applies_floors_per_source_before_fusion(self):
        """An RRF score is a rank artefact with no notion of 'relevant
        enough', so a post-fusion filter could not express this."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[])
        engine._keyword_candidates = MagicMock(return_value=[])

        engine.search(
            "q", top_k=5, mode=HYBRID,
            thresholds={"min_vector_score": 0.6, "min_keyword_score": 0.05},
        )

        assert engine._vector_candidates.call_args.args[2] == 0.6
        assert engine._keyword_candidates.call_args.args[2] == 0.05

    def test_rerank_floor_can_empty_the_result(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._reranker.rerank.return_value = [chunk(1, score=-8.0)]

        resp = engine.search(
            "q", top_k=5, rerank=True, mode=SEMANTIC,
            thresholds={"min_rerank_score": 0.0},
        )

        assert resp.results == []

    def test_rerank_floor_keeps_confident_hits(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._reranker.rerank.return_value = [chunk(1, score=5.0)]

        resp = engine.search(
            "q", top_k=5, rerank=True, mode=SEMANTIC,
            thresholds={"min_rerank_score": 0.0},
        )

        assert [r.chunk_id for r in resp.results] == [1]

    def test_default_floors_match_the_measured_curve(self):
        """Only the rerank floor is on by default, and only because it was
        measured free: abstention 0.000 -> 0.500 at identical recall. The
        bi-encoder floors stay off - their score distributions for relevant
        and irrelevant queries overlap, so any floor there costs recall."""
        from config import Config

        c = Config().search
        assert c.min_vector_score is None
        assert c.min_keyword_score is None
        assert c.min_rerank_score == -8.0

    def test_default_mode_is_hybrid(self):
        from config import Config

        assert Config().search.mode == "hybrid"


class TestWeightedFusion:
    """Plain RRF weights sources equally, which assumes they're comparably
    good rankers. Ours aren't - keyword alone scores nDCG 0.598 vs dense's
    0.787, and equal weighting measured worse than dense alone."""

    def test_weights_default_to_equal(self):
        fused = rrf_fuse([[chunk(1)], [chunk(2)]], k=60)
        assert fused[0]["score"] == fused[1]["score"]

    def test_down_weighted_source_cannot_outvote_the_stronger_one(self):
        """A document the strong ranker put 20th, that the weak ranker put
        first. Unweighted, the weak source's vote is enough to promote it over
        the strong ranker's own top hit - which is the dilution measured on
        the eval set. Down-weighted, it can't."""
        strong = [chunk(i) for i in range(1, 21)]
        weak = [chunk(20)]

        assert rrf_fuse([strong, weak], k=60)[0]["chunk_id"] == 20
        assert rrf_fuse([strong, weak], k=60, weights=[1.0, 0.1])[0]["chunk_id"] == 1

    def test_weight_scales_contribution_linearly(self):
        fused = rrf_fuse([[chunk(1)]], k=60, weights=[0.5])
        assert fused[0]["score"] == pytest.approx(0.5 / 61)

    def test_zero_weight_still_contributes_the_document(self):
        """Weight 0 must not silently drop a source's documents - they're
        still candidates, just with no rank credit from that list."""
        fused = rrf_fuse([[chunk(1)], [chunk(2)]], k=60, weights=[1.0, 0.0])
        assert {c["chunk_id"] for c in fused} == {1, 2}
        assert fused[0]["chunk_id"] == 1

    def test_hybrid_passes_the_configured_weight(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])

        cfg = MagicMock()
        cfg.search.mode = "hybrid"
        cfg.search.rrf_k = 60
        cfg.search.hybrid_candidates = 20
        cfg.search.keyword_weight = 0.1
        cfg.search.min_vector_score = None
        cfg.search.min_keyword_score = None
        cfg.search.min_rerank_score = None
        with patch("config.load", return_value=cfg):
            resp = engine.search("q", top_k=5, mode=HYBRID)

        # the down-weighted keyword hit must rank below the dense hit
        assert [r.chunk_id for r in resp.results] == [1, 2]


class TestRerankerScore:
    def test_rerank_reports_the_cross_encoder_score_not_the_old_one(self):
        """Regression: the result dict was built as {"score": new, **chunk},
        and chunk already carries a "score" - so the spread overwrote the
        cross-encoder score with the pre-rerank one. Ordering was unaffected
        (it sorts on the raw scores), but the reported number was wrong,
        which made it useless to threshold on or show."""
        from unittest.mock import MagicMock as MM

        from process.embedder import Reranker

        r = Reranker.__new__(Reranker)  # skip loading a real model
        r._model = MM()
        r._model.predict.return_value = [0.9, 0.1]

        out = r.rerank("q", [chunk(1, score=0.0164), chunk(2, score=0.0161)])

        assert [c["chunk_id"] for c in out] == [1, 2]
        assert out[0]["score"] == pytest.approx(0.9)
        assert out[1]["score"] == pytest.approx(0.1)

    def test_reranked_flag_stays_true_when_the_floor_empties_results(self):
        """An empty result after a floor means 'reranked, nothing cleared the
        bar' - not 'never reranked'. Deriving the flag from emptiness reported
        the opposite of what happened."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._reranker.rerank.return_value = [chunk(1, score=-9.0)]

        resp = engine.search(
            "q", rerank=True, mode=SEMANTIC, thresholds={"min_rerank_score": -8.0}
        )

        assert resp.results == []
        assert resp.reranked is True


class TestRerankCandidatePool:
    """#27 — reranking only pays off when the cross-encoder gets more
    candidates than it returns. Retrieving 5 and reranking to 5 reorders those
    5 and can never promote a 6th, which is what shipped."""

    def test_pool_widens_retrieval_but_not_the_result(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(
            return_value=[chunk(i) for i in range(1, 41)]
        )
        engine._reranker.rerank.return_value = [chunk(7), chunk(3)]

        resp = engine.search(
            "q", top_k=2, rerank=True, rerank_top_k=2, mode=SEMANTIC, candidates=40
        )

        # retrieval asked for the pool...
        engine._vector_candidates.assert_called_once_with("q", 40, None)
        # ...the reranker saw all of it...
        assert len(engine._reranker.rerank.call_args.args[1]) == 40
        # ...and the caller still got what it asked for.
        assert [r.chunk_id for r in resp.results] == [7, 3]

    def test_a_chunk_ranked_below_top_k_can_now_win(self):
        """The whole point: chunk 12 is invisible to a top_k=3 retrieval, and
        reachable once the pool is 40 wide."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(
            return_value=[chunk(i) for i in range(1, 41)]
        )
        engine._reranker.rerank.side_effect = lambda q, chunks, top_k: sorted(
            chunks, key=lambda c: 0 if c["chunk_id"] == 12 else 1
        )[:top_k]

        resp = engine.search(
            "q", top_k=3, rerank=True, rerank_top_k=3, mode=SEMANTIC, candidates=40
        )

        assert resp.results[0].chunk_id == 12

    def test_pool_is_ignored_without_reranking(self):
        """Nothing would narrow 40 back down to top_k, so the caller would get
        40 results it never asked for."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])

        engine.search("q", top_k=5, rerank=False, mode=SEMANTIC, candidates=40)

        engine._vector_candidates.assert_called_once_with("q", 5, None)

    def test_pool_never_narrows_an_already_wider_request(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._reranker.rerank.return_value = [chunk(1)]

        engine.search(
            "q", top_k=50, rerank=True, rerank_top_k=5, mode=SEMANTIC, candidates=40
        )

        engine._vector_candidates.assert_called_once_with("q", 50, None)

    def test_omitting_the_pool_keeps_the_old_exact_top_k_behaviour(self):
        """eval.sweep and the /search endpoint measure retrieval itself, so
        top_k must keep meaning "retrieve exactly this many" for them."""
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._reranker.rerank.return_value = [chunk(1)]

        engine.search("q", top_k=5, rerank=True, rerank_top_k=5, mode=SEMANTIC)

        engine._vector_candidates.assert_called_once_with("q", 5, None)

    def test_hybrid_pool_covers_both_sources_before_fusion(self):
        engine = make_engine()
        engine._vector_candidates = MagicMock(return_value=[chunk(1)])
        engine._keyword_candidates = MagicMock(return_value=[chunk(2)])
        engine._reranker.rerank.return_value = [chunk(1)]

        engine.search(
            "q", top_k=5, rerank=True, rerank_top_k=5, mode=HYBRID, candidates=40
        )

        # both sources pull the full pool - fusing a wide list with a narrow
        # one would let the narrow source cap what fusion has to work with
        assert engine._vector_candidates.call_args.args[1] == 40
        assert engine._keyword_candidates.call_args.args[1] == 40


class _FakeSession:
    """Stands in for a SQLAlchemy session over a chunks table held in a dict
    keyed by (obj_id, chunk_index). Only expand_neighbours' one query shape
    is supported, which is the point - it asserts on that shape."""

    def __init__(self, rows: dict):
        self._rows = rows
        self.queries = 0
        self.asked_for: set = set()

    def execute(self, stmt):
        self.queries += 1
        wanted = set(stmt.whereclause.right.value)
        self.asked_for |= wanted
        return SimpleNamespace(
            all=lambda: [
                SimpleNamespace(obj_id=o, chunk_index=i, chunk_text=text)
                for (o, i), text in sorted(self._rows.items())
                if (o, i) in wanted
            ]
        )


class TestExpandNeighbours:
    """#29 — HybridChunker emits no overlap, so a sentence spanning a chunk
    boundary is split and neither half reads as an answer."""

    def test_glues_the_window_around_the_hit(self):
        session = _FakeSession(
            {(1, 3): "before.", (1, 4): "the hit.", (1, 5): "after."}
        )

        [got] = expand_neighbours(session, [chunk(4, chunk_index=4, obj_id=1)], window=1)

        assert got["context"] == "before.\n\nthe hit.\n\nafter."

    def test_matched_chunk_text_is_left_alone(self):
        """`score` refers to `text`. Widening it in place would make the score
        look like it applied to all three chunks."""
        session = _FakeSession(
            {(1, 3): "before.", (1, 4): "the hit.", (1, 5): "after."}
        )

        [got] = expand_neighbours(
            session, [chunk(4, chunk_index=4, obj_id=1, text="the hit.")], window=1
        )

        assert got["text"] == "the hit."

    def test_document_edges_close_up_rather_than_leaving_blanks(self):
        session = _FakeSession({(1, 0): "first chunk.", (1, 1): "second."})

        [got] = expand_neighbours(session, [chunk(0, chunk_index=0, obj_id=1)], window=1)

        assert got["context"] == "first chunk.\n\nsecond."

    def test_never_crosses_into_another_pdf(self):
        """chunk_index restarts per object, so matching on index alone would
        splice one paper's text into another's."""
        session = _FakeSession(
            {(1, 4): "paper one.", (2, 3): "paper two before.", (2, 5): "paper two after."}
        )

        [got] = expand_neighbours(session, [chunk(4, chunk_index=4, obj_id=1)], window=1)

        assert got["context"] == "paper one."

    def test_widening_the_window_pulls_more(self):
        session = _FakeSession({(1, i): f"c{i}." for i in range(6)})

        [got] = expand_neighbours(session, [chunk(3, chunk_index=3, obj_id=1)], window=2)

        assert got["context"] == "c1.\n\nc2.\n\nc3.\n\nc4.\n\nc5."

    def test_one_query_regardless_of_hit_count(self):
        """A query per hit is 5-10 round trips on the request path."""
        session = _FakeSession({(1, i): f"c{i}." for i in range(20)})
        hits = [chunk(i, chunk_index=i, obj_id=1) for i in (2, 7, 12, 17)]

        expand_neighbours(session, hits, window=1)

        assert session.queries == 1

    def test_window_zero_is_a_no_op(self):
        session = _FakeSession({(1, 4): "x"})
        hits = [chunk(4, chunk_index=4, obj_id=1)]

        assert expand_neighbours(session, hits, window=0) is hits
        assert session.queries == 0

    def test_empty_results_never_hit_the_database(self):
        session = _FakeSession({})

        assert expand_neighbours(session, [], window=1) == []
        assert session.queries == 0

    def test_never_asks_for_a_negative_index(self):
        """A hit at index 0 has no chunk -1. Sending impossible keys would
        still return the right answer, so this checks the IN clause itself
        rather than the result."""
        session = _FakeSession({(1, 0): "first."})

        expand_neighbours(session, [chunk(0, chunk_index=0, obj_id=1)], window=2)

        assert session.asked_for == {(1, 0), (1, 1), (1, 2)}
