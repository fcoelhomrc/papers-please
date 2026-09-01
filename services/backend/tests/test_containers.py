"""Tests for reading worker container state and logs (#38).

No container runtime: the HTTP layer is stubbed. What's under test is the
name matching, the log de-framing, and the two behaviours that matter when
things are wrong — a missing worker and an unreachable runtime.
"""
import struct
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api
import containers


def _frame(stream: int, payload: bytes) -> bytes:
    """A Docker log frame: stream id, three zero bytes, big-endian length."""
    return bytes([stream, 0, 0, 0]) + struct.pack(">I", len(payload)) + payload


def _stub_client(containers_json=None, log_body=b""):
    client = MagicMock()
    client.__enter__ = lambda s: s
    client.__exit__ = lambda s, *a: None

    def get(url, params=None):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if url.endswith("/containers/json"):
            response.json.return_value = containers_json or []
        else:
            response.content = log_body
        return response

    client.get.side_effect = get
    return client


class TestDemux:
    def test_strips_frame_headers(self):
        raw = _frame(1, b"line one\n") + _frame(2, b"line two\n")

        assert containers.demux(raw) == "line one\nline two\n"

    def test_interleaves_stdout_and_stderr_in_order(self):
        raw = _frame(1, b"out\n") + _frame(2, b"err\n") + _frame(1, b"out again\n")

        assert containers.demux(raw) == "out\nerr\nout again\n"

    def test_passes_through_an_unframed_tty_stream(self):
        """A TTY-allocated container sends no framing. Parsing it as frames
        would mangle real output."""
        assert containers.demux(b"plain log output\n") == "plain log output\n"

    def test_empty_body(self):
        assert containers.demux(b"") == ""

    def test_invalid_utf8_does_not_raise(self):
        raw = _frame(1, b"caf\xff\n")

        assert "caf" in containers.demux(raw)


class TestListWorkers:
    def _run(self, containers_json):
        with (
            patch.object(containers, "_client", return_value=_stub_client(containers_json)),
        ):
            return containers.list_workers()

    def test_matches_podman_compose_names(self):
        workers = self._run(
            [{"Names": ["/papers-please_worker-download_1"], "State": "running", "Status": "Up 2 hours"}]
        )
        by_service = {w["service"]: w for w in workers}

        assert by_service["worker-download"]["state"] == "running"
        assert by_service["worker-download"]["status"] == "Up 2 hours"

    def test_matches_compose_v2_names(self):
        """Separators differ by tool: `papers-please_worker-chunk_1` under
        podman-compose, `papers-please-worker-chunk-1` under compose v2."""
        workers = self._run(
            [{"Names": ["/papers-please-worker-chunk-1"], "State": "exited", "Status": "Exited (1)"}]
        )
        by_service = {w["service"]: w for w in workers}

        assert by_service["worker-chunk"]["state"] == "exited"

    def test_a_service_with_no_container_is_missing_not_omitted(self):
        """'This worker has never been started' is the single most useful
        thing this endpoint can say — it is exactly the state that produced
        the stuck-downloads report. Omitting it leaves the UI blank, which
        is what it already did."""
        workers = self._run([])

        assert [w["state"] for w in workers] == ["missing"] * 3
        assert [w["service"] for w in workers] == list(containers.WORKER_SERVICES)

    def test_ignores_a_same_named_worker_from_another_project(self):
        """The runtime lists every container on the host, and a dev box
        routinely runs several compose stacks. Matching on the service name
        alone would report another project's worker as ours."""
        workers = self._run(
            [
                {"Names": ["/other-stack_worker-download_1"], "State": "running", "Status": "Up"},
                {"Names": ["/papers-please_db_1"], "State": "running", "Status": "Up"},
            ]
        )

        assert [w["state"] for w in workers] == ["missing"] * 3

    def test_respects_COMPOSE_PROJECT_NAME(self, monkeypatch):
        monkeypatch.setenv("COMPOSE_PROJECT_NAME", "other-stack")
        workers = self._run(
            [{"Names": ["/other-stack_worker-download_1"], "State": "running", "Status": "Up"}]
        )
        by_service = {w["service"]: w for w in workers}

        assert by_service["worker-download"]["state"] == "running"

    def test_reports_every_service_in_pipeline_order(self):
        workers = self._run([])

        assert [w["service"] for w in workers] == [
            "worker-download",
            "worker-chunk",
            "worker-embed",
        ]


class TestWorkerLogs:
    def test_rejects_a_name_that_is_not_a_known_worker(self):
        """Without this the endpoint reads logs from any container on the
        host, since the runtime lists them all."""
        with pytest.raises(ValueError, match="unknown worker"):
            containers.worker_logs("papers-please_db_1")

    def test_returns_nothing_for_a_worker_with_no_container(self):
        with patch.object(containers, "_client", return_value=_stub_client([])):
            assert containers.worker_logs("worker-download") == ""

    def test_returns_demuxed_output(self):
        stub = _stub_client(
            [{"Names": ["/papers-please_worker-embed_1"], "State": "running", "Status": "Up"}],
            log_body=_frame(1, b"Embedded 12/12 chunks\n"),
        )
        with patch.object(containers, "_client", return_value=stub):
            assert containers.worker_logs("worker-embed") == "Embedded 12/12 chunks\n"


class TestWorkersEndpoint:
    def test_an_unreachable_runtime_is_not_an_error(self, monkeypatch):
        """Running the backend outside compose is a normal setup. Reporting
        it as workers-are-down would be a worse lie than saying nothing."""
        monkeypatch.setattr(
            containers, "list_workers", MagicMock(side_effect=containers.RuntimeUnavailable("no socket"))
        )
        body = TestClient(api.app).get("/workers").json()

        assert body["workers"] == []
        assert "no socket" in body["unavailable"]

    def test_reports_worker_state(self, monkeypatch):
        monkeypatch.setattr(
            containers,
            "list_workers",
            MagicMock(
                return_value=[
                    {
                        "service": "worker-download",
                        "container": "papers-please_worker-download_1",
                        "state": "exited",
                        "status": "Exited (1) 5 minutes ago",
                        "exit_code": 1,
                    }
                ]
            ),
        )
        body = TestClient(api.app).get("/workers").json()

        assert body["unavailable"] is None
        assert body["workers"][0]["state"] == "exited"
        assert body["workers"][0]["exit_code"] == 1

    def test_logs_for_an_unknown_worker_are_a_404(self, monkeypatch):
        monkeypatch.setattr(
            containers, "worker_logs", MagicMock(side_effect=ValueError("unknown worker 'x'"))
        )

        assert TestClient(api.app).get("/workers/x/logs").status_code == 404

    def test_logs_when_the_runtime_is_down_are_a_503(self, monkeypatch):
        monkeypatch.setattr(
            containers,
            "worker_logs",
            MagicMock(side_effect=containers.RuntimeUnavailable("no socket")),
        )

        assert TestClient(api.app).get("/workers/worker-chunk/logs").status_code == 503
