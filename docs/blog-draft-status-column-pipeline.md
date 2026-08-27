---
category:
  - "[[Notes]]"
created_at: 2026-08-27
topics:
  - "[[Technical Portfolio]]"
created_by:
  - "[[AI Generated]]"
---
# Why our ingestion pipeline is a status column, not a queue

*Draft design note.*

Papers Please fetches paper metadata from Semantic Scholar, downloads the
PDF, OCRs and chunks it, embeds the chunks, and indexes them for search.
Five steps, each slower than the last, each able to fail on a PDF that's
corrupt, paywalled, or just too big. Something has to track what's done and
what still needs doing.

This note records what we considered, what we picked, and what we
knowingly gave up — at the stage the project is at today, before any
agent sits in front of it.

## 1. What we looked at

**A message broker or task queue (Celery, RQ, pgmq).** The standard
answer for "process N things async, retry the failures." We ruled it out
for the same reason a lot of small pipelines do: it's a second system to
run and reason about, and our actual concurrency need is modest — one
box, a handful of PDFs in flight, not a fleet of workers competing for
work. Bringing in a broker to coordinate a single process felt like
solving a problem we don't have yet.

**A workflow engine (Airflow, Prefect).** Same story, more so. These want
a DAG definition, a scheduler, a UI. Our "DAG" is four steps in a fixed
order with no branching. Adopting one now would mean spending more code on
describing the pipeline than the pipeline itself contains.

**A status column plus a polling loop.** Postgres already holds every
document, PDF object, and chunk. Give each row a `status`, have one loop
ask "what's pending?" and do it. No new infrastructure, no dual-write
problem — the same transaction that finishes a step also updates the row
that says so.

## 2. What we picked, and why we're comfortable

We picked the status column.

`objects.status` is `pending | chunked | failed`. Each stage's job is a
tight loop shape: query rows in the state it acts on, do the work, write
the next state.

```python
class PdfChunker(PostgresInterface):
    def pending(self) -> list[tuple[int, str]]:
        # SELECT id, path FROM objects WHERE status = 'pending'
        ...

    def process(self, obj_id: int, path: str):
        try:
            chunks = self._chunk_pdf(path)
            self._write_chunks(obj_id, chunks)   # status -> 'chunked', same transaction
        except Exception as e:
            self._mark_failed(obj_id)            # status -> 'failed'
```

`PdfFetcher`, `PdfChunker`, and `PdfEmbedder` all follow this shape
independently — download, chunk, and embed each own their pending-query,
their write, and their failure path. Nothing but a shared `status` string
couples them. The pipeline order (fetch → download → chunk → embed) is
encoded once, in `worker.py`, as a sequence of calls — not as a schema, not
as a scheduler config.

```python
def run():
    PdfFetcher(...).execute()
    PdfChunker().execute()
    PdfEmbedder().execute()
```

A cron-shaped loop wraps that: run the three stages, sleep, repeat. It's
the smallest thing that's correct today, and because each stage only knows
its own status transition, splitting them into separate processes later —
which the roadmap calls for — doesn't touch this contract at all.

## 3. How it actually works

### Failure is a status, not an exception that escapes

Every stage catches its own failures and writes `status = 'failed'` rather
than letting an exception kill the loop or silently drop the row:

```python
def _mark_failed(self, obj_id: int):
    session.execute(update(Object).where(Object.id == obj_id).values(status="failed"))
```

One bad PDF doesn't stop the batch — the loop moves to the next pending
row. And `failed` isn't a dead end: once a full sweep finds nothing left
`pending`, failed rows are requeued automatically:

```python
def execute(self, limit=None):
    for obj_id, path in self.pending()[:limit]:
        self.process(obj_id, path)
    if not self.pending():          # sweep is clear
        self._requeue_failed()      # failed -> pending, try again next cycle
```

This is a deliberately dumb retry policy — no backoff, no attempt cap — but
it costs nothing to reason about, and a corrupt PDF that fails forever just
keeps getting relogged, which is enough to notice it in the logs.

### Reconciliation instead of transactional writes

PDFs land on disk and get registered in Postgres as two separate
operations (`save()`, then `register()`), not one transaction — there's no
way to atomically write a file and commit a database row together. A
process killed between the two leaves an orphan: a file with no row, or a
row with no file.

Rather than trying to make that atomic, there's a `reconcile()` pass that
runs the comparison after the fact: delete stray files with no matching
row, delete rows whose file is missing or fails a magic-byte check, clean
up `.tmp` leftovers from interrupted downloads.

```python
def reconcile(self):
    # ... remove orphaned .tmp files, corrupt/missing objects, and stray PDFs
```

This is the same idea as a lease timeout in a task-queue design — accept
that a crash mid-operation is going to happen, and design the *next* run
to clean up after it, instead of trying to prevent it from ever happening.

### Idempotent writes at the boundary

Every stage's write is an upsert with `ON CONFLICT DO NOTHING`:

```python
session.execute(
    insert(Chunk).on_conflict_do_nothing(index_elements=["obj_id", "chunk_index"]),
    rows,
)
```

Combined with polling on `status`, this means running a stage twice on the
same row — because a poll cycle overlapped, or a retry re-ran — is a
no-op, not a duplicate. The queue doesn't need "exactly once" delivery
guarantees, because the writes are already safe to apply more than once.

## 4. What we gave up

**One worker, one process.** Today's `worker.py` runs all three stages in
one loop on one machine. There's no competing-consumers pattern, no
`SKIP LOCKED`-style claiming — because there's only ever one reader of
`pending`. That's fine at today's scale (a handful of papers at a time)
and is exactly the coupling the project's own `docs/mvp-plan.md` flags as
the first thing to fix: splitting each stage into its own process, still
reading the same `status` column, still no new infrastructure.

**No backoff, no priority, no per-item observability beyond `status` and
logs.** A failing PDF retries at the same cadence as everything else,
forever, with no signal beyond "still failing" in the log stream. Given
how small the failure surface is at this scale (malformed PDFs, dead
URLs), that's an acceptable place to not build tooling yet.

**The pipeline order lives in code, not data.** `worker.py`'s three-line
`run()` *is* the DAG. That's the right amount of structure for four fixed
steps in a fixed order — but it's also exactly what stops being true once
an orchestrator needs to *decide* what runs next instead of being told.
That's next.
