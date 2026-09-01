"""Tests for the download stage's failure handling (#37).

No network and no database: `download()` is exercised against a stubbed
`requests.get`, and the attempt bookkeeping against a stubbed session. The
retry cap mirrors the shape already covered for chunking in
tests/test_chunker_text.py::TestRetryCap.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingest.fetcher import MIN_PDF_BYTES, PdfFetcher


def _response(status=200, body=b"%PDF-1.4" + b"x" * MIN_PDF_BYTES, content_type="application/pdf"):
    return SimpleNamespace(
        status_code=status,
        content=body,
        headers={"content-type": content_type} if content_type else {},
    )


class TestDownloadValidation:
    """A 200 is not enough. Many open-access links resolve to an HTML
    landing page, and some publishers serve one with a 200 - which used to
    be written as `<source_id>.pdf`, registered as a real object, and handed
    to Docling, burning the chunker's whole retry budget on OCR of a web
    page."""

    def _download(self, response):
        # tries=1 so a rejection surfaces immediately instead of retrying
        with patch("ingest.fetcher.requests.get", return_value=response):
            return PdfFetcher.download.__wrapped__("http://example.invalid/x.pdf")

    def test_accepts_a_real_pdf(self):
        body = b"%PDF-1.4" + b"x" * MIN_PDF_BYTES

        assert self._download(_response(body=body)) == body

    def test_rejects_an_html_landing_page_served_as_200(self):
        with pytest.raises(ValueError, match="not a PDF"):
            self._download(_response(body=b"<html>...", content_type="text/html"))

    def test_rejects_a_landing_page_mislabelled_as_a_pdf(self):
        """Servers lie in both directions, so the magic bytes decide and the
        header is only used to make the error readable."""
        with pytest.raises(ValueError, match="%PDF"):
            self._download(_response(body=b"<!DOCTYPE html>", content_type="application/pdf"))

    def test_accepts_an_octet_stream_that_is_really_a_pdf(self):
        """The other direction of the same lie: a real PDF served as
        octet-stream must not be rejected on the header alone."""
        body = b"%PDF-1.7" + b"x" * MIN_PDF_BYTES

        assert self._download(_response(body=body, content_type="application/octet-stream")) == body

    def test_rejects_a_truncated_pdf(self):
        with pytest.raises(ValueError, match="truncated"):
            self._download(_response(body=b"%PDF-1.4 tiny"))

    def test_a_non_200_still_raises(self):
        with pytest.raises(requests.HTTPError):
            self._download(_response(status=403, body=b"<html>", content_type="text/html"))


class _Obj:
    def __init__(self, status="downloading", attempts=0, path="x.pdf"):
        self.status = status
        self.attempts = attempts
        self.path = path


def _fetcher(existing=None, max_attempts=3):
    f = PdfFetcher.__new__(PdfFetcher)  # skip __init__ (config, storage)
    f.max_attempts = max_attempts
    f.engine = MagicMock()

    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = lambda s, *a: None
    session.execute.return_value = MagicMock(
        scalar_one_or_none=MagicMock(return_value=existing)
    )
    f._session = session
    return f


def _run(fetcher, fn, *args, **kwargs):
    import ingest.fetcher as mod

    with patch.object(mod, "Session", return_value=fetcher._session):
        return fn(*args, **kwargs)


class TestAttemptBookkeeping:
    def test_first_attempt_creates_the_row_before_the_request(self):
        """Counting after the fact loses the attempt whenever the process
        dies mid-download, and an uncounted attempt is an infinite retry
        with extra steps."""
        f = _fetcher(existing=None)

        _run(f, f._begin_attempt, 7, "paper.pdf")

        added = f._session.add.call_args.args[0]
        assert added.status == "downloading"
        assert added.attempts == 1
        assert added.doc_id == 7

    def test_a_retry_increments_rather_than_inserting_again(self):
        obj = _Obj(status="failed", attempts=1)
        f = _fetcher(existing=obj)

        _run(f, f._begin_attempt, 7, "paper.pdf")

        assert obj.attempts == 2
        assert obj.status == "downloading"
        f._session.add.assert_not_called()

    def test_success_hands_the_object_to_the_chunker(self):
        """'pending' is PdfChunker.pending()'s entry condition."""
        obj = _Obj(attempts=1)
        f = _fetcher(existing=obj)

        _run(f, f._finish_attempt, 7, ok=True)

        assert obj.status == "pending"

    def test_a_failure_under_the_cap_stays_retryable(self):
        obj = _Obj(attempts=1)
        f = _fetcher(existing=obj, max_attempts=3)

        _run(f, f._finish_attempt, 7, ok=False)

        assert obj.status == "downloading"

    def test_reaching_the_cap_gives_up(self):
        """The bug this replaces: a DOI link that 403s forever was retried
        on every pass, ~28 requests/min at doi.org, with nothing recorded."""
        obj = _Obj(attempts=3)
        f = _fetcher(existing=obj, max_attempts=3)

        _run(f, f._finish_attempt, 7, ok=False)

        assert obj.status == "failed"

    def test_a_lower_cap_gives_up_sooner(self):
        obj = _Obj(attempts=1)
        f = _fetcher(existing=obj, max_attempts=1)

        _run(f, f._finish_attempt, 7, ok=False)

        assert obj.status == "failed"

    def test_a_vanished_row_is_not_an_error(self):
        f = _fetcher(existing=None)

        _run(f, f._finish_attempt, 7, ok=True)  # must not raise


class TestTask:
    def _fetcher_for_task(self, tmp_path, download):
        f = _fetcher(existing=_Obj())
        f.store_root = tmp_path
        f._tmp_dir = tmp_path / ".tmp"
        f.download = download
        return f

    def test_a_successful_download_saves_and_marks_pending(self, tmp_path):
        body = b"%PDF-1.4" + b"x" * MIN_PDF_BYTES
        f = self._fetcher_for_task(tmp_path, MagicMock(return_value=body))

        ok = _run(f, f.task, 7, "http://example.invalid/x.pdf", "paper.pdf")

        assert ok is True
        assert (tmp_path / "paper.pdf").read_bytes() == body

    def test_a_failed_download_writes_no_file_but_records_the_attempt(self, tmp_path):
        obj = _Obj(attempts=2)
        f = self._fetcher_for_task(tmp_path, MagicMock(side_effect=ValueError("not a PDF")))
        f._session.execute.return_value = MagicMock(
            scalar_one_or_none=MagicMock(return_value=obj)
        )
        f.max_attempts = 3

        ok = _run(f, f.task, 7, "http://example.invalid/x", "paper.pdf")

        assert ok is False
        assert not (tmp_path / "paper.pdf").exists()
        # begin bumped 2 -> 3, finish then hit the cap
        assert obj.attempts == 3
        assert obj.status == "failed"
