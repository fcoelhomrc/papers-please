"""arXiv-only fetching for the eval corpus.

The corpus needs a ~100% PDF hit rate, and Semantic Scholar's own
openAccessPdf.url does not deliver it: for most arXiv papers it points at ACL
Anthology, MDPI, or a doi.org landing page that answers 403 to a scripted
fetch. That is what left six documents stuck at 'downloading' before.

These tests pin the two things that make the hit rate hold - the URL is
synthesised from the arXiv id rather than trusted, and the budget counts
papers kept rather than papers seen - plus the staging rule that keeps
unpromoted candidates out of the OCR queue.
"""
from unittest.mock import MagicMock, patch

from ingest.fetcher import FIELDS, SemanticScholarFetcher
from ingest.schemas import DocumentTemplate, arxiv_pdf_url


class TestArxivPdfUrl:
    def test_synthesises_a_direct_pdf_url_from_the_id(self):
        assert arxiv_pdf_url({"externalIds": {"ArXiv": "2305.14251"}}) == (
            "https://arxiv.org/pdf/2305.14251"
        )

    def test_none_without_an_arxiv_id(self):
        assert arxiv_pdf_url({"externalIds": {"DOI": "10.1/x"}}) is None

    def test_none_when_external_ids_is_missing_entirely(self):
        assert arxiv_pdf_url({}) is None


class TestDocumentTemplateFromS2:
    def test_arxiv_url_beats_the_open_access_url(self):
        """The recorded reason this matters: in one sampled page of 1000
        results, 187 papers had an arXiv id and only 73 had an arxiv.org PDF
        URL. Trusting openAccessPdf.url reintroduces the doi.org 403s."""
        doc = DocumentTemplate.from_s2(
            {
                "paperId": "p1",
                "title": "T",
                "externalIds": {"ArXiv": "2305.14251"},
                "openAccessPdf": {"url": "https://doi.org/10.1/x"},
            }
        )

        assert doc.pdf_url == "https://arxiv.org/pdf/2305.14251"

    def test_falls_back_to_open_access_url_without_an_arxiv_id(self):
        doc = DocumentTemplate.from_s2(
            {
                "paperId": "p1",
                "title": "T",
                "openAccessPdf": {"url": "https://aclanthology.org/x.pdf"},
            }
        )

        assert doc.pdf_url == "https://aclanthology.org/x.pdf"

    def test_carries_corpus_and_topic(self):
        doc = DocumentTemplate.from_s2(
            {"paperId": "p1", "title": "T"}, corpus="candidate", topic="retrieval"
        )

        assert (doc.corpus, doc.topic) == ("candidate", "retrieval")

    def test_defaults_to_the_main_corpus(self):
        """Papers fetched through the UI must not land in the eval corpus."""
        assert DocumentTemplate.from_s2({"paperId": "p", "title": "T"}).corpus == "main"

    def test_carries_citation_count(self):
        doc = DocumentTemplate.from_s2(
            {"paperId": "p1", "title": "T", "citationCount": 42}
        )

        assert doc.citation_count == 42


class TestFields:
    def test_requests_external_ids_and_citation_count(self):
        """Without externalIds there is no arXiv id to synthesise from, and
        the fetch silently degrades to the unreliable URLs."""
        assert "externalIds" in FIELDS and "citationCount" in FIELDS


class TestFetchParameters:
    def _fetch(self, **kwargs):
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        with (
            patch.object(fetcher, "_paginate", return_value=iter([])) as paginate,
            patch.object(fetcher, "_write", return_value=0),
        ):
            fetcher.fetch(query="x", **kwargs)
        return paginate.call_args.args[0]

    def test_open_access_pdf_is_sent_as_a_valueless_flag(self):
        """`openAccessPdf=true` is rejected by the endpoint; the bare key is
        what the API expects."""
        assert self._fetch(open_access_pdf=True)["openAccessPdf"] == ""

    def test_omits_the_flag_when_not_requested(self):
        assert "openAccessPdf" not in self._fetch()

    def test_forwards_min_citations_and_sort(self):
        params = self._fetch(min_citations=10, sort="citationCount:desc")

        assert (params["minCitationCount"], params["sort"]) == (10, "citationCount:desc")


class TestArxivOnlyFiltering:
    def test_drops_papers_without_an_arxiv_id(self):
        """Filtered client-side: the API has no arXiv filter, and `venue` is
        not a usable proxy because arXiv papers carry inconsistent venue
        strings."""
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        batch = [
            {"paperId": "a", "externalIds": {"ArXiv": "1"}},
            {"paperId": "b", "externalIds": {"DOI": "x"}},
            {"paperId": "c", "externalIds": None},
        ]

        with (
            patch.object(fetcher, "_paginate", return_value=iter([batch])),
            patch.object(fetcher, "_write", return_value=1) as write,
        ):
            fetcher.fetch(query="x", max_papers=10, arxiv_only=True)

        assert [d["paperId"] for d in write.call_args.args[0]] == ["a"]

    def test_budget_counts_papers_kept_not_papers_seen(self):
        """Otherwise a topic where most results are non-arXiv would stop
        early and come back short of its quota."""
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)
        mixed = [
            {"paperId": f"a{i}", "externalIds": {"ArXiv": str(i)}} for i in range(2)
        ] + [{"paperId": f"b{i}", "externalIds": {}} for i in range(8)]

        with (
            patch.object(fetcher, "_paginate", return_value=iter([mixed, mixed])),
            patch.object(fetcher, "_write", return_value=2) as write,
        ):
            fetcher.fetch(query="x", max_papers=4, arxiv_only=True)

        assert write.call_count == 2

    def test_passes_corpus_and_topic_to_the_writer(self):
        fetcher = SemanticScholarFetcher.__new__(SemanticScholarFetcher)

        with (
            patch.object(
                fetcher, "_paginate", return_value=iter([[{"paperId": "a"}]])
            ),
            patch.object(fetcher, "_write", return_value=1) as write,
        ):
            fetcher.fetch(query="x", corpus="candidate", topic="agents")

        assert write.call_args.kwargs == {"corpus": "candidate", "topic": "agents"}


class TestCandidatesAreNotDownloaded:
    def test_pending_excludes_the_candidate_corpus(self):
        """Downloading 200 PDFs to keep 100 wastes hours of OCR on papers
        that were never going to be in the corpus."""
        from ingest.fetcher import PdfFetcher

        fetcher = PdfFetcher.__new__(PdfFetcher)
        fetcher.max_attempts = 3
        fetcher.engine = MagicMock()
        session = MagicMock()
        session.execute.return_value.all.return_value = []

        with patch("ingest.fetcher.Session") as sess:
            sess.return_value.__enter__.return_value = session
            fetcher.pending()

        assert "documents.corpus != " in str(session.execute.call_args.args[0])


class TestTopics:
    def test_five_topics_with_unique_slugs(self):
        """Few-and-deep: within-topic neighbours are what make retrieval
        hard, and twenty thin topics also collapse the multi-hop
        synthesizers."""
        from eval.corpus import TOPICS

        assert len(TOPICS) == 5 and len({s for s, _ in TOPICS}) == 5

    def test_over_fetches_to_leave_the_reviewer_a_choice(self):
        from eval.corpus import CANDIDATES_PER_TOPIC, KEEP_PER_TOPIC

        assert CANDIDATES_PER_TOPIC >= 2 * KEEP_PER_TOPIC

    def test_stage_requests_arxiv_only_candidates(self):
        from eval import corpus

        fetcher = MagicMock()
        fetcher.fetch.return_value = 0
        with patch("ingest.fetcher.SemanticScholarFetcher", return_value=fetcher):
            corpus.stage_candidates(per_topic=40)

        kwargs = fetcher.fetch.call_args.kwargs
        assert kwargs["arxiv_only"] and kwargs["corpus"] == "candidate"
