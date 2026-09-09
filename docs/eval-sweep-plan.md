# Finishing the retrieval evaluation

Written 2026-09-09. **Frozen scope — three steps, no additions.**

---

## 1. Finish extracting embeddings — DONE

All four encoders at 11,381 / 11,381 chunks.

| model | embedded |
|---|---|
| bge-small | 11,381 |
| bge-large | 11,381 |
| arctic-m-v2 | 11,381 |
| qwen3-0.6b | 11,381 |

---

## 2. Free branch ablations, including reranker, on everything

`all` = `a` (mode × top_k) + `b` (rerank) + `c` (query arms) + `w`
(keyword_weight × rrf_k) + `pool` + `timings`.

| model | status |
|---|---|
| bge-small | done — `ablation-all-20260907T005619Z.json` |
| bge-large | **only `a`** — needs `all` |
| arctic-m-v2 | **nothing** — needs `all` |
| qwen3-0.6b | **nothing** — needs `all` |

Three runs. For each, set `embedder.model` in `config.yaml`, then:

```bash
podman-compose run -d -T --name papers-please_ablate-<model> \
  worker-embed python -m eval.ablations all
podman update --restart=no papers-please_ablate-<model>
```

Cost: **$0**. Roughly 1.5–2h each, ~5h total, unattended.

---

## 3. LLM-as-judge with the different techniques

Six query arms, matching what bge-small already has: `none`,
`multi_query`, `multi_query+orig`, `decompose`, `decompose+orig`, `hyde`.

| target | arms done | arms remaining |
|---|---|---|
| bge-small | all 6 | — |
| bge-large | `none` | 5 |
| arctic-m-v2 | `none` | 5 |
| qwen3-0.6b | `none` | 5 |
| BM25 | `none` | 5 |

**20 runs remaining.** For each, set `embedder.model` in `config.yaml`
(irrelevant for BM25, which uses no encoder), then:

```bash
podman-compose run -d -T -e PAPERS_PLEASE_REPLAY= \
  --name papers-please_judged-<target>-<arm> \
  backend python -m eval.run --mode <semantic|bm25> --arm <arm>
podman update --restart=no papers-please_judged-<target>-<arm>
```

`--mode semantic` for the three encoders, `--mode bm25` for BM25.

Cost: **~$0.19 per run, ~$3.80 total** (tonight's four ran $0.176–$0.207).
About 30m each, ~10h serially.

---

## Totals

| | time | cost |
|---|---|---|
| 2. free ablations × 3 | ~5h | $0 |
| 3. judged × 20 | ~10h | ~$3.80 |
| | **~15h** | **~$3.80** |

---

## Flags both commands need

- `-e PAPERS_PLEASE_REPLAY=` — `.env` sets it to `1`, which puts the answerer
  on the recorded cassette and returns 100 empty answers.
- `podman update --restart=no` after launching — `podman-compose run -d`
  inherits the service's `restart: unless-stopped` and will loop the container
  on exit, success or failure.
- Test suite runs on the host, not in the container:
  `cd services/backend && uv run pytest tests/ -q`.
