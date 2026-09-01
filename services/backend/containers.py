"""Reading worker container state and logs from the container runtime.

The pipeline stages run as separate processes with no HTTP surface, so the
backend cannot ask them how they are. It can ask the thing that started
them. Podman exposes a Docker-compatible API over a unix socket, and httpx
already speaks unix sockets, so this needs no new dependency and no changes
to the workers themselves.

The motivating incident: six papers sat at "downloading" and were reported
as a broken pipeline. The pipeline was fine - the download worker simply
was not running, and nothing in the UI could say so. A stopped worker and a
busy one looked identical.

Read-only on purpose. Starting and stopping containers from a web request
is a much larger security surface than reading their status, and restarting
a worker is one compose command.
"""
import logging
import os
import re
import struct

import httpx

logger = logging.getLogger(__name__)

# Compose service names, in pipeline order.
WORKER_SERVICES = ("worker-download", "worker-chunk", "worker-embed")

# Compose derives the container name prefix from the project. Matching on
# the service name alone would pick up a same-named worker from an
# unrelated project on the same machine, which is not hypothetical - a dev
# box routinely has several compose stacks on one runtime.
DEFAULT_PROJECT = "papers-please"

DEFAULT_SOCKET = f"/run/user/{os.getuid()}/podman/podman.sock"
API_VERSION = "v1.41"

# Docker frames each log chunk with 8 bytes: [stream][000][big-endian length].
_FRAME_HEADER = 8
_STREAMS = (0, 1, 2)  # stdin, stdout, stderr


class RuntimeUnavailable(RuntimeError):
    """The container runtime could not be reached.

    Entirely normal - the backend may be running outside compose, or the
    podman socket may not be enabled - so callers render this as "unknown"
    rather than as a failure.
    """


def socket_path() -> str:
    """`DOCKER_HOST` if set (compose sets it to a unix:// URL), else the
    rootless podman default."""
    host = os.environ.get("DOCKER_HOST", "")
    if host.startswith("unix://"):
        return host[len("unix://") :]
    return host or DEFAULT_SOCKET


def _client() -> httpx.Client:
    path = socket_path()
    if not os.path.exists(path):
        raise RuntimeUnavailable(f"no container runtime socket at {path}")
    # base_url's host is ignored for a unix transport but httpx requires one.
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=path), base_url="http://runtime", timeout=5.0
    )


def _normalise(name: str) -> str:
    """Compose names differ by tool and version - `papers-please_worker-chunk_1`
    under podman-compose, `papers-please-worker-chunk-1` under compose v2 -
    so separators are flattened before matching."""
    return re.sub(r"[-_]+", "-", name.strip("/").lower())


def project() -> str:
    return os.environ.get("COMPOSE_PROJECT_NAME") or DEFAULT_PROJECT


def _service_of(container_names: list[str]) -> str | None:
    """Which worker service a container belongs to, if any.

    Requires both the project prefix and the service name, so a
    `worker-download` belonging to some other stack on the same runtime is
    not mistaken for ours.
    """
    prefix = _normalise(project())
    for raw in container_names:
        normalised = _normalise(raw)
        if not normalised.startswith(prefix):
            continue
        for service in WORKER_SERVICES:
            if _normalise(service) in normalised:
                return service
    return None


def list_workers() -> list[dict]:
    """One entry per known worker service, whether or not it exists.

    A service with no container at all is reported as `missing` rather than
    omitted: "this worker has never been started" is the single most useful
    thing this endpoint can say, and it is exactly the state that produced
    the incident above. Omitting it would leave the UI showing nothing,
    which is what it already did.
    """
    with _client() as client:
        response = client.get(
            f"/{API_VERSION}/containers/json", params={"all": "true"}
        )
        response.raise_for_status()
        containers = response.json()

    # The runtime lists every container on the host, including other
    # projects'. Only the ones matching a known service are considered.
    found: dict[str, dict] = {}
    for container in containers:
        service = _service_of(container.get("Names") or [])
        if service and service not in found:
            found[service] = {
                "service": service,
                "container": (container.get("Names") or ["?"])[0].lstrip("/"),
                "state": container.get("State", "unknown"),
                "status": container.get("Status", ""),
                "exit_code": container.get("ExitCode"),
            }

    return [
        found.get(
            service,
            {
                "service": service,
                "container": None,
                "state": "missing",
                "status": "not created",
                "exit_code": None,
            },
        )
        for service in WORKER_SERVICES
    ]


def demux(raw: bytes) -> str:
    """Strip Docker's stream framing from a log body.

    Without a TTY the runtime interleaves stdout and stderr, prefixing every
    chunk with 8 bytes: a stream id, three zero bytes, then a big-endian
    length. Rendering that verbatim puts binary garbage between every line.

    A TTY-allocated container sends the stream unframed, so anything that
    doesn't parse as a frame is passed through as-is rather than mangled.
    """
    out: list[bytes] = []
    i = 0
    while i + _FRAME_HEADER <= len(raw):
        stream = raw[i]
        if stream not in _STREAMS or raw[i + 1 : i + 4] != b"\x00\x00\x00":
            return raw.decode("utf-8", "replace")  # not framed
        (length,) = struct.unpack(">I", raw[i + 4 : i + _FRAME_HEADER])
        i += _FRAME_HEADER
        out.append(raw[i : i + length])
        i += length
    if not out:
        return raw.decode("utf-8", "replace")
    return b"".join(out).decode("utf-8", "replace")


def worker_logs(service: str, tail: int = 200) -> str:
    """Recent output from one worker.

    `service` must be one of WORKER_SERVICES. That is a real constraint, not
    tidiness: the runtime lists every container on the machine, so an
    unvalidated name would turn this endpoint into a log reader for anything
    else running on the host.
    """
    if service not in WORKER_SERVICES:
        raise ValueError(f"unknown worker {service!r}")

    worker = next(w for w in list_workers() if w["service"] == service)
    if not worker["container"]:
        return ""

    with _client() as client:
        response = client.get(
            f"/{API_VERSION}/containers/{worker['container']}/logs",
            params={"stdout": "true", "stderr": "true", "tail": str(tail)},
        )
        response.raise_for_status()
        return demux(response.content)
