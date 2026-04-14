import logging
import os
from pathlib import Path

from docling.chunking import HybridChunker  # type: ignore
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from transformers import AutoTokenizer

from db.connection import PostgresInterface
from db.models import Chunk, Object

logger = logging.getLogger(__name__)

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 512

OBJECT_STORE_ROOT = os.environ.get("OBJECT_STORE_ROOT", "data")


class PdfChunker(PostgresInterface):
    def __init__(self, store_root: str = OBJECT_STORE_ROOT):
        super().__init__()
        self.store_root = store_root
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
            max_tokens=MAX_TOKENS,
        )
        self._converter = DocumentConverter()
        self._chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, repeat_table_header=True)

    def pending(self) -> list[tuple[int, str]]:
        stmt = select(Object.id, Object.path).where(Object.status == "pending")
        with Session(self.engine) as session:
            rows = session.execute(stmt).all()
        logger.info(f"{len(rows)} objects pending chunking")
        return [(r.id, r.path) for r in rows]

    def _chunk_pdf(self, path: Path) -> list[tuple[int, str, int | None]]:
        doc = self._converter.convert(source=str(path)).document
        result = []
        for i, chunk in enumerate(self._chunker.chunk(dl_doc=doc)):
            page_num = None
            if chunk.meta and chunk.meta.doc_items:
                prov = chunk.meta.doc_items[0].prov
                if prov:
                    page_num = prov[0].page_no
            if chunk.text:
                result.append((i, chunk.text, page_num))
        return result

    def _write_chunks(self, obj_id: int, chunks: list[tuple[int, str, int | None]]):
        rows = [
            {"obj_id": obj_id, "chunk_index": idx, "chunk_text": text, "page_num": page}
            for idx, text, page in chunks
        ]
        with Session(self.engine) as session:
            session.execute(
                insert(Chunk).on_conflict_do_nothing(index_elements=["obj_id", "chunk_index"]),
                rows,
            )
            session.execute(update(Object).where(Object.id == obj_id).values(status="chunked"))
            session.commit()

    def _mark_failed(self, obj_id: int):
        with Session(self.engine) as session:
            session.execute(update(Object).where(Object.id == obj_id).values(status="failed"))
            session.commit()

    def process(self, obj_id: int, path: str):
        pdf_path = Path(self.store_root) / path
        try:
            chunks = self._chunk_pdf(pdf_path)
            self._write_chunks(obj_id, chunks)
            logger.info(f"Chunked object {obj_id}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to chunk object {obj_id} ({path}): {e}")
            self._mark_failed(obj_id)

    def execute(self):
        for obj_id, path in self.pending():
            self.process(obj_id, path)
