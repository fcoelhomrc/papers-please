"""The /search endpoint's mode plumbing - that the query param actually
reaches the engine, and that an invalid one is rejected at the edge rather
than blowing up inside retrieval."""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api
from schemas import SearchResponse


def client_with_engine(engine):
    api.app.dependency_overrides[api.get_engine] = lambda: engine
    return TestClient(api.app)


def make_engine(mode="hybrid"):
    engine = MagicMock()
    engine.search.return_value = SearchResponse(
        query="q", model="bge-small", mode=mode, reranked=False, results=[]
    )
    return engine


def teardown_function():
    api.app.dependency_overrides.clear()


def test_mode_is_passed_through_to_the_engine():
    engine = make_engine()
    resp = client_with_engine(engine).get("/search", params={"q": "robots", "mode": "hybrid"})

    assert resp.status_code == 200
    assert engine.search.call_args.kwargs["mode"] == "hybrid"
    assert resp.json()["mode"] == "hybrid"


def test_omitted_mode_defers_to_config():
    """No mode in the request must reach the engine as None, so config decides
    - not as a hardcoded 'semantic' that would shadow the setting."""
    engine = make_engine("semantic")
    client_with_engine(engine).get("/search", params={"q": "robots"})

    assert engine.search.call_args.kwargs["mode"] is None


def test_unknown_mode_rejected_with_422():
    engine = make_engine()
    resp = client_with_engine(engine).get("/search", params={"q": "x", "mode": "magic"})

    assert resp.status_code == 422
    engine.search.assert_not_called()


def test_each_valid_mode_is_accepted():
    for mode in ("semantic", "keyword", "hybrid"):
        engine = make_engine(mode)
        resp = client_with_engine(engine).get("/search", params={"q": "x", "mode": mode})
        assert resp.status_code == 200, mode
