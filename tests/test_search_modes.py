
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
