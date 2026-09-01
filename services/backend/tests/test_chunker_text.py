"""Tests for what actually gets embedded (#28): the heading path prefix, the
boilerplate filter, and the vector metadata that rides along.

Pure functions plus one stubbed chunk loop - no docling conversion, no PDF,
no Pinecone, no network.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from process.chunker import (
    PdfChunker,
    contextualize,
    heading_path,
    is_boilerplate,
)
from process.embedder import chunk_metadata


class TestHeadingPath:
    def test_joins_the_hierarchy(self):
        assert heading_path(["Methods", "Ablation Studies"]) == "Methods > Ablation Studies"

    def test_single_heading(self):
        assert heading_path(["Introduction"]) == "Introduction"

    def test_no_headings_is_empty(self):
        assert heading_path(None) == ""
        assert heading_path([]) == ""

    def test_drops_blank_entries_rather_than_emitting_stray_separators(self):
        assert heading_path(["Methods", "  ", "Setup"]) == "Methods > Setup"

    def test_strips_surrounding_whitespace(self):
        assert heading_path([" Results \n"]) == "Results"


class TestContextualize:
    def test_prefixes_the_path(self):
        assert contextualize("We ablate the encoder.", ["Methods", "Ablations"]) == (
            "Methods > Ablations\n\nWe ablate the encoder."
        )

    def test_text_is_unchanged_without_headings(self):
        assert contextualize("Body text.", None) == "Body text."

    def test_body_text_survives_verbatim(self):
        """The prefix is added, never a rewrite - this string is what gets
        embedded, displayed on result cards and handed to the LLM."""
        body = "Table 1 | acc 0.91 | f1 0.88"

        assert contextualize(body, ["Results"]).endswith(body)


class TestIsBoilerplate:
    @pytest.mark.parametrize(
        "heading",
        [
            "References",
            "REFERENCES",
            "references and notes",
            "Bibliography",
            "Acknowledgments",
            "Acknowledgements",
            "Author Contributions",
            "Competing Interests",
            "Conflict of Interest Statement",
            "Funding",
            "Supplementary Material",
        ],
    )
    def test_drops_sections_with_no_retrievable_claims(self, heading):
        assert is_boilerplate([heading]) is True

    @pytest.mark.parametrize(
        "heading",
        ["6. References", "A.2 Acknowledgments", "VII - Bibliography", "3) Funding"],
    )
    def test_sees_through_publisher_numbering(self, heading):
        assert is_boilerplate([heading]) is True

    @pytest.mark.parametrize(
        "heading",
        [
            "Introduction",
            "Methods",
            "Results",
            "Reference implementation",  # a real Methods subsection
            "Discussion",
        ],
    )
    def test_keeps_real_content(self, heading):
        assert is_boilerplate([heading]) is False

    def test_judges_the_deepest_heading_not_an_ancestor(self):
        """"Appendix > Ablation results" is a real result a question can be
        about; only the section the chunk actually sits in decides."""
        assert is_boilerplate(["Appendix", "Ablation results"]) is False
        assert is_boilerplate(["Discussion", "Acknowledgments"]) is True

    def test_no_headings_is_kept(self):
        """Body text docling couldn't attribute to a section is still body
        text - dropping it would silently lose abstracts and front matter."""
        assert is_boilerplate(None) is False
        assert is_boilerplate([]) is False


def _chunk(text, headings=None, page=1):
    prov = [SimpleNamespace(page_no=page)] if page is not None else []
    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(headings=headings, doc_items=[SimpleNamespace(prov=prov)]),
    )


def _chunker_with(chunks):
    """A PdfChunker whose docling converter/chunker are stubbed out."""
    chunker = PdfChunker.__new__(PdfChunker)  # skip __init__ (docling, DB, models)
    chunker._converter = MagicMock()
    chunker._chunker = MagicMock()
    chunker._chunker.chunk.return_value = chunks
    return chunker


class TestChunkPdf:
    def test_stores_heading_prefixed_text_with_its_page(self):
        chunker = _chunker_with([_chunk("We ablate.", ["Methods", "Ablations"], page=4)])

        assert chunker._chunk_pdf("x.pdf") == [
            (0, "Methods > Ablations\n\nWe ablate.", 4)
        ]

    def test_boilerplate_never_reaches_the_index(self):
        chunker = _chunker_with(
            [
                _chunk("Real finding.", ["Results"]),
                _chunk("[1] Smith et al...", ["References"]),
                _chunk("We thank the reviewers.", ["Acknowledgments"]),
            ]
        )

        texts = [text for _, text, _ in chunker._chunk_pdf("x.pdf")]

        assert texts == ["Results\n\nReal finding."]

    def test_chunk_index_stays_dense_across_dropped_chunks(self):
        """chunk_index is half the (obj_id, chunk_index) key and is what
        neighbour expansion walks. Numbering by position in docling's output
        would leave a hole wherever boilerplate was dropped, and "the chunk
        before this one" would sometimes be nothing at all."""
        chunker = _chunker_with(
            [
                _chunk("A", ["Intro"]),
                _chunk("[1] Smith...", ["References"]),
                _chunk("B", ["Methods"]),
                _chunk("We thank...", ["Acknowledgements"]),
                _chunk("C", ["Results"]),
            ]
        )

        assert [i for i, _, _ in chunker._chunk_pdf("x.pdf")] == [0, 1, 2]

    def test_empty_chunks_are_skipped(self):
        chunker = _chunker_with([_chunk("", ["Intro"]), _chunk("Real.", ["Intro"])])

        assert len(chunker._chunk_pdf("x.pdf")) == 1

    def test_missing_provenance_yields_no_page(self):
        chunker = _chunker_with([_chunk("Body.", ["Intro"], page=None)])

        assert chunker._chunk_pdf("x.pdf")[0][2] is None


class TestChunkMetadata:
    def test_carries_the_filterable_fields(self):
        assert chunk_metadata(page_num=3, doc_id=7, year=2024) == {
            "page_num": 3,
            "doc_id": 7,
            "year": 2024,
        }

    def test_unset_fields_are_omitted_not_nulled(self):
        """Pinecone rejects null metadata values, and a sentinel like 0 would
        match a `year >= 2023` filter the wrong way round."""
        assert chunk_metadata(page_num=None, doc_id=7, year=None) == {"doc_id": 7}

    def test_all_unset_is_an_empty_dict(self):
        assert chunk_metadata(page_num=None, year=None) == {}


class TestNormaliseHeading:
    """The normaliser is what lets one entry per section cover the dozen ways
    a publisher might print it - and what must not over-reach."""

    def test_strips_numbering_punctuation_and_case(self):
        from process.chunker import normalise_heading

        assert normalise_heading("6.  REFERENCES:") == "references"

    def test_leaves_a_word_that_merely_starts_with_a_numeral_letter(self):
        """Regression: a roman-numeral character class ate the leading C of
        'Competing Interests', normalising it to 'ompeting interests' and
        silently keeping a boilerplate section in the index."""
        from process.chunker import normalise_heading

        assert normalise_heading("Competing Interests") == "competing interests"
        assert normalise_heading("Introduction") == "introduction"
        assert normalise_heading("Validation") == "validation"

    def test_drops_publisher_filler_suffixes(self):
        from process.chunker import normalise_heading

        assert normalise_heading("Conflict of Interest Statement") == "conflict of interest"
        assert normalise_heading("References and Notes") == "references"

    def test_does_not_truncate_a_real_heading_to_a_boilerplate_name(self):
        from process.chunker import normalise_heading

        assert normalise_heading("Reference implementation") == "reference implementation"


class TestRetryCap:
    """#35 — without a cap, _requeue_failed flipped every failure back to
    pending whenever the queue emptied, so a PDF that will never parse was
    re-OCR'd forever. Requeueing also kept the queue non-empty, which is the
    condition that fires the method: one broken file could hold the chunker
    at 100% CPU indefinitely."""

    def _chunker(self, failed_rows, max_attempts=3):
        from unittest.mock import MagicMock

        chunker = PdfChunker.__new__(PdfChunker)
        chunker.max_attempts = max_attempts
        chunker.engine = MagicMock()

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = lambda s, *a: None
        session.execute.return_value = MagicMock(all=MagicMock(return_value=failed_rows))
        chunker._session = session
        return chunker

    def _requeue(self, chunker):
        from unittest.mock import patch

        import process.chunker as mod

        updates = []
        original_execute = chunker._session.execute

        def execute(stmt):
            compiled = str(stmt)
            if compiled.strip().upper().startswith("UPDATE"):
                updates.append(stmt.compile().params)
                return MagicMock()
            return original_execute(stmt)

        chunker._session.execute = execute
        with patch.object(mod, "Session", return_value=chunker._session):
            chunker._requeue_failed()
        return updates

    def test_a_retryable_failure_goes_back_to_pending(self):
        chunker = self._chunker([(1, 0)])

        [update] = self._requeue(chunker)

        assert update["status"] == "pending"

    def test_a_repeat_offender_is_marked_dead_not_pending(self):
        """'failed' means try again, 'dead' means stop trying. Collapsing
        them loses the ability to tell a transient OCR failure from a PDF
        that will never parse."""
        chunker = self._chunker([(1, 3)])

        [update] = self._requeue(chunker)

        assert update["status"] == "dead"

    def test_the_boundary_is_the_configured_attempt_count(self):
        chunker = self._chunker([(1, 2), (2, 3)], max_attempts=3)

        updates = self._requeue(chunker)

        assert [u["status"] for u in updates] == ["pending", "dead"]

    def test_a_lower_cap_gives_up_sooner(self):
        chunker = self._chunker([(1, 1)], max_attempts=1)

        [update] = self._requeue(chunker)

        assert update["status"] == "dead"
