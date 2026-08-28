"""Unit test for GET /status."""
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


def test_status_returns_pipeline_status():
    fake_status = {
        "documents_total": 42,
        "pending_download": 5,
        "objects_by_status": {"pending": 2, "chunked": 10},
        "chunks_pending_embed": 3,
        "embed_model": "BAAI/bge-small-en-v1.5",
    }
    with patch("api.pipeline_status", return_value=fake_status):
        client = TestClient(api.app)
        response = client.get("/status")

    assert response.status_code == 200
    assert response.json() == fake_status
