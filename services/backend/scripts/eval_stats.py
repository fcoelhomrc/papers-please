"""Descriptive statistics for the curated eval set.

    uv run python -m scripts.eval_stats

The numbers that belong in the README and in any writeup of a result: how big
the haystack is, how the questions are distributed, and - the one people leave
out - the precision ceiling, since precision@k is capped at min(R,k)/k and a
raw precision figure reads as failure without it.
"""
import json
from collections import Counter
from pathlib import Path
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from db.connection import PostgresInterface
from db.models import Chunk, Document, Object

D = Path("eval/testset")
cur = [json.loads(l) for l in (D / "curated.jsonl").read_text().splitlines() if l.strip()]
gen = [json.loads(l) for l in (D / "generated.jsonl").read_text().splitlines() if l.strip()]
judged = {}
for l in (D / "judged.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); judged[r["id"]] = r

ids = sorted({c for q in cur for c in q["reference_chunk_ids"]})
with Session(PostgresInterface.connect()) as s:
    et = dict(s.execute(select(Chunk.id, Chunk.element_type).where(Chunk.id.in_(ids))).all())
    n_chunks = s.scalar(select(func.count()).select_from(Chunk))
    n_docs = s.scalar(select(func.count()).select_from(Document).where(Document.corpus == "eval"))
    n_obj = s.scalar(select(func.count()).select_from(Object))

def line(k, v): print(f"  {k:<34} {v}")

print("\nCORPUS (the haystack)")
line("papers", n_docs); line("PDFs on disk", n_obj); line("chunks", f"{n_chunks:,}")
line("chunks per question", f"{n_chunks / len(cur):.0f}x")

print("\nQUESTION SET")
line("generated -> curated", f"{len(gen)} -> {len(cur)}")
line("edited during curation", sum(1 for q in cur if q["edited"]))
for k, v in sorted(Counter(q["topics"][0] for q in cur).items()):
    line(f"  topic: {k}", v)
for k, v in sorted(Counter(q["synthesizer"].replace("_query_synthesizer", "") for q in cur).items()):
    line(f"  type: {k}", f"{v}  ({v/len(cur):.0%})")

print("\nLABELS")
gold = [len(q["reference_chunk_ids"]) for q in cur]
line("questions with 0 gold chunks", sum(1 for g in gold if g == 0))
line("gold chunks: min/median/max", f"{min(gold)} / {sorted(gold)[len(gold)//2]} / {max(gold)}")
line("total gold chunk labels", sum(gold))
line("distinct gold chunks", len(ids))
line("distinct papers covered", len({d for q in cur for d in q["reference_doc_ids"]}))
for k, v in sorted(Counter(gold).items()):
    line(f"  questions with {k} gold chunk(s)", v)

print("\nPRECISION CEILING (min(R,k)/k, the best precision@k this set allows)")
for k in (1, 5, 10, 20):
    line(f"  precision@{k} ceiling", f"{sum(min(g, k)/k for g in gold)/len(gold):.3f}")

print("\nGOLD CHUNK STRUCTURE")
kinds = Counter(et.get(c) for c in ids)
for k, v in kinds.most_common():
    line(f"  {k}", f"{v:>4}  ({v/len(ids):.0%})")

print("\nQUESTION SURFACE")
lens = sorted(len(q["question"]) for q in cur)
line("chars: min/median/max", f"{lens[0]} / {lens[len(lens)//2]} / {lens[-1]}")
line("over 300 chars", sum(1 for x in lens if x > 300))
scores = [judged[q["id"]].get("score", 0) for q in cur if q["id"] in judged]
line("curation score: min/median/max", f"{min(scores)} / {sorted(scores)[len(scores)//2]} / {max(scores)}")
print()
