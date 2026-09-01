import logging
import re
from pathlib import Path

from docling.chunking import HybridChunker  # type: ignore
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from db.connection import PostgresInterface
from db.models import Chunk, Object
from process.embedder import MODELS

logger = logging.getLogger(__name__)

# Sections that are structurally present in almost every paper and carry no
# retrievable claims. Dropping them is not just index hygiene: a references
# section is a dense list of author surnames and title fragments, so it
# matches the keyword retriever for practically any author or topic query,
# and scores high ts_rank doing it because the terms are packed. That's a
# whole class of confident, useless keyword hits removed at the source.
#
# Matched as a whole normalised heading, never as a prefix: "Reference
# implementation" is a real Methods subsection and starts with "reference".
# Deliberately excludes "Appendix" - appendices routinely hold ablations,
# proofs and extra results that questions are actually about.
BOILERPLATE_SECTIONS = frozenset(
    {
        "reference",
        "references",
        "bibliography",
        "works cited",
        "acknowledgment",
        "acknowledgments",
        "acknowledgement",
        "acknowledgements",
        "author contribution",
        "author contributions",
        "author information",
        "competing interest",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
        "declaration of competing interest",
        "declaration of interest",
        "funding",
        "funding information",
        "disclosure",
        "disclosures",
        "data availability",
        "code availability",
        "supplementary material",
        "supplementary materials",
        "about the authors",
    }
)

# Leading enumeration docling keeps from the PDF's own numbering: "6.",
# "A.2", "VII -", "3)". Required to be followed by whitespace, which is what
# stops it eating the "C" of "Competing Interests" as a roman numeral.
_ENUMERATION = re.compile(
    r"^\s*(?:\d+|[ivxlcIVXLC]+|[A-Za-z])(?:[.\-]\d+)*(?:\s*[.):\-\u2013\u2014])?\s+"
)

# Suffixes publishers bolt onto an otherwise standard section name.
_TRAILING_FILLER = ("statements", "statement", "and notes", "information", "section")


def normalise_heading(heading: str) -> str:
    """Strip numbering, punctuation and publisher filler down to the bare
    section name, so one entry in BOILERPLATE_SECTIONS covers the dozen ways
    a journal might print it."""
    text = _ENUMERATION.sub("", heading).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    for filler in _TRAILING_FILLER:
        if text.endswith(f" {filler}"):
            text = text[: -len(filler) - 1].strip()
    return text


def is_boilerplate(headings: list[str] | None) -> bool:
    """Whether a chunk sits under a section with no retrievable content.

    Judged on the deepest heading only. A subsection of Acknowledgments is
    still acknowledgments, but "Appendix > Ablation results" is a real result
    a question could be about - so an ancestor being boilerplate does not
    condemn the child.

    Unheaded text is kept: docling leaves abstracts and front matter without
    a heading, and dropping those would lose the most quotable part of a
    paper to a tidy-up rule.
    """
    if not headings:
        return False
    return normalise_heading(headings[-1]) in BOILERPLATE_SECTIONS


def heading_path(headings: list[str] | None) -> str:
    """"Methods > Ablation Studies" - the chunk's position in the paper."""
    return " > ".join(h.strip() for h in headings if h and h.strip()) if headings else ""


def contextualize(text: str, headings: list[str] | None) -> str:
    """The text that actually gets embedded and shown.

    docling hands us the body text with its section headings stripped out
    into metadata, and we were storing only the body - so two papers' Methods
    sections describing the same technique were near-identical vectors with
    nothing to tell them apart, and a query naming a section ("ablation
    results") had no term to match.

    Prepending costs nothing against the token budget: HybridChunker sizes
    chunks by counting `contextualize()` on its own metadata-joined form,
    which already includes these headings - so storing bare text was leaving
    the 512-token window under-filled, not saving room in it.

    Own formatting rather than docling's `chunker.contextualize()`, which
    joins metadata with newlines: this string is displayed verbatim on search
    result cards and in the agent's context, and one " > " line reads as a
    breadcrumb where a stack of bare newlines reads as part of the passage.
    """
    path = heading_path(headings)
    return f"{path}\n\n{text}" if path else text


class PdfChunker(PostgresInterface):
    def __init__(self, store_root: str | None = None):
        from config import load

        super().__init__()
        config = load()
        self.store_root = store_root or config.storage.root
        device = (
            AcceleratorDevice.CPU
            if config.devices.chunker == "cpu"
            else AcceleratorDevice.CUDA
        )
        self.max_attempts = config.stages.chunk.max_attempts
        pipeline_options = PdfPipelineOptions(ocr_options=RapidOcrOptions())
        pipeline_options.accelerator_options = AcceleratorOptions(device=device)
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                MODELS[config.embedder.model]["hf_name"]
            ),
            max_tokens=config.embedder.max_tokens,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self._chunker = HybridChunker(
            tokenizer=tokenizer, merge_peers=True, repeat_table_header=True
        )

    def pending(self) -> list[tuple[int, str]]:
        stmt = select(Object.id, Object.path).where(Object.status == "pending")
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        logger.info(f"{len(rows)} objects pending chunking")
        return [(r.id, r.path) for r in rows]

    def failed(self) -> list[tuple[int, str]]:
        stmt = select(Object.id, Object.path).where(Object.status == "failed")
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        logger.info(f"{len(rows)} objects failed chunking")
        return [(r.id, r.path) for r in rows]

    def _chunk_pdf(self, path: Path) -> list[tuple[int, str, int | None]]:
        doc = self._converter.convert(source=str(path)).document
        result = []
        # `i` counts kept chunks, not chunks seen: chunk_index is half of the
        # (obj_id, chunk_index) key and is what neighbour expansion walks, so
        # it has to stay dense. Numbering by position in docling's output
        # would leave holes wherever boilerplate was dropped, and "the chunk
        # before this one" would then sometimes be nothing at all.
        for chunk in self._chunker.chunk(dl_doc=doc):
            if not chunk.text:
                continue

            headings = chunk.meta.headings if chunk.meta else None
            if is_boilerplate(headings):
                continue

            page_num = None
            if chunk.meta and chunk.meta.doc_items:
                prov = chunk.meta.doc_items[0].prov
                if prov:
                    page_num = prov[0].page_no

            result.append((len(result), contextualize(chunk.text, headings), page_num))
        return result

    def _write_chunks(self, obj_id: int, chunks: list[tuple[int, str, int | None]]):
        rows = [
            {"obj_id": obj_id, "chunk_index": idx, "chunk_text": text, "page_num": page}
            for idx, text, page in chunks
        ]
        with Session(self.engine) as session:
            session.execute(
                insert(Chunk).on_conflict_do_nothing(
                    index_elements=["obj_id", "chunk_index"]
                ),
                rows,
            )
            session.execute(
                update(Object).where(Object.id == obj_id).values(status="chunked")
            )
            session.commit()

    def _mark_failed(self, obj_id: int):
        """Record the failure and count the attempt.

        The attempt is counted here rather than at requeue time so a crash
        between the two doesn't lose it - an uncounted attempt is an infinite
        retry with extra steps.
        """
        with Session(self.engine) as session:
            session.execute(
                update(Object)
                .where(Object.id == obj_id)
                .values(status="failed", attempts=Object.attempts + 1)
            )
            session.commit()

    def _requeue_failed(self):
        """Give failed objects another go - up to a point.

        Without the cap this flipped every failure back to pending whenever
        the queue emptied, so a PDF that will never parse was re-OCR'd
        forever. Worse, requeueing kept the queue non-empty, which is the
        condition that fires this method: one broken file could keep the
        chunker at 100% CPU indefinitely.

        Past the cap the object goes 'dead' rather than 'failed' - not the
        same thing. 'failed' means "try again", 'dead' means "stop trying",
        and only the second is a state the loop can safely leave alone.
        """
        requeued = dead = 0
        with Session(self.engine) as session:
            rows = session.execute(
                select(Object.id, Object.attempts).where(Object.status == "failed")
            ).all()
            for obj_id, attempts in rows:
                status = "pending" if attempts < self.max_attempts else "dead"
                session.execute(
                    update(Object).where(Object.id == obj_id).values(status=status)
                )
                if status == "pending":
                    requeued += 1
                else:
                    dead += 1
            session.commit()

        logger.info(
            f"{requeued} objects requeued for chunking, {dead} gave up after "
            f"{self.max_attempts} attempts"
        )

    def process(self, obj_id: int, path: str):
        pdf_path = Path(self.store_root) / path
        try:
            chunks = self._chunk_pdf(pdf_path)
            self._write_chunks(obj_id, chunks)
            logger.info(f"Chunked object {obj_id}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to chunk object {obj_id} ({path}): {e}")
            self._mark_failed(obj_id)

    def execute(self, limit: int | None = None):
        pending = self.pending()

        if limit:
            pending = pending[:limit]
        for obj_id, path in pending:
            self.process(obj_id, path)

        pending = self.pending()
        if len(pending) == 0:
            self._requeue_failed()
