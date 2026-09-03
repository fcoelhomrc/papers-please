# Visualising the ragas knowledge graph and test set

Written 2026-09-04. A study of the ragas codebase and a proposed contribution.

**Not being built here.** Preserving question→node provenance changes what
`TestsetSample` records at generation time, so adopting it would mean
regenerating the test set. This corpus is generated and reviewed; the note
exists so the findings survive, not as a work item.

---

## Why the question came up

Nothing in ragas lets you look at what it built. `KnowledgeGraph` exposes six
methods — `add`, `find_indirect_clusters`, `find_two_nodes_single_rel`,
`load`, `remove_node`, `save` — and none of them show you anything. There is
no plotting code, no HTML, no SVG, and no templating anywhere in the package.

That mattered concretely during generation. Three of the four failures in
`docs/rag-evaluation.md` were invisible until they became expensive:

- a cosine threshold of 0.5 admitted **94.7% of all possible node pairs**,
  making a complete graph whose depth-3 cluster search never returns — it hung
  a run for 13 minutes at 0% CPU before being killed
- **zero `entities_overlap` relationships** existed, because one failed NER
  call left a node without `entities` and `OverlapScoreBuilder` aborts on the
  first pair involving it — silently removing every multi-hop-specific
  question from a test set that still looked complete
- the generated set was **skewed to one topic** (75 of 131 questions), because
  ragas seeds by graph centrality rather than by topic

Each is a property of the graph, obvious in one glance at the right summary,
and undetectable from the API as it stands.

---

## What constrains a contribution

### Dependency posture

This is the binding constraint, and it is stricter than most projects.

Core dependencies are `numpy`, `datasets`, `tiktoken`, `langchain`,
`langchain-core`, `langchain-community`, `langchain_openai`, `nest-asyncio`,
`appdirs`, `pydantic>=2`, `openai`, `diskcache`. Extras are `all`, `docs`,
`dev`.

**`pandas` is not core** — it sits in the `all` extra, and `to_pandas()` is
guarded (`dataset_schema.py:236`). So is `rapidfuzz`, which is why
`OverlapScoreBuilder` raises `ImportError` at construction without it. There
is no plotting library, no `networkx`, no `jinja2`, and no `IPython` anywhere.

A project that declines to make pandas a hard dependency will not take a
graph-drawing library.

### Conventions worth matching

| observation | where | implication |
|---|---|---|
| `to_list` / `to_csv` / `to_jsonl` / `to_pandas` / `to_hf_dataset` | `dataset_schema.py` | export is a method on the data object, not a separate module |
| `save()` / `load()` → plain JSON with a UUID encoder | `testset/graph.py:179,203` | serialisation is already dependency-free |
| **no `_repr_html_`, no IPython import** | package-wide | notebook rich-display would be a *new* convention |
| **no CLI, no entry points** | `entry_points.txt` absent | a `ragas viz` command would be the first |
| `try: import X / except ImportError: raise ImportError("X is not installed. Please install it to use this function.")` | 12 sites | optional deps are allowed, in exactly this shape |
| `tqdm.auto` for progress | `async_utils.py` | notebook-aware without depending on IPython |

---

## The core finding: provenance is destroyed

`BaseScenario` (`testset/synthesizers/base.py:43`) carries everything a
visualisation needs:

```python
class BaseScenario(BaseModel):
    nodes: t.List[Node]
    style: QueryStyle
    length: QueryLength
    persona: Persona
```

`TestsetSample` (`testset/synthesizers/testset_schema.py:22`) keeps two fields:

```python
class TestsetSample(BaseSample):
    eval_sample: t.Union[SingleTurnSample, MultiTurnSample]
    synthesizer_name: str
```

And at `testset/synthesizers/generate.py:441-445` the scenario is in scope
while exactly one field is recorded:

```python
additional_testset_info.append({"synthesizer_name": synthesizer.name})
```

So there is **no supported way to ask which nodes produced a question.** The
persona that asked it, the style and length it was generated at, and the graph
nodes it came from are all discarded.

### What that costs in practice

This repo reconstructs the link by byte-exact matching of
`reference_contexts` against node `page_content`, which requires:

- feeding ragas our own chunks as nodes, and passing `transforms=` explicitly
  so no `HeadlineSplitter` ever re-cuts them — `default_transforms` inserts one
  when ≥25% of documents exceed 500 tokens
- stripping the `<1-hop>` / `<2-hop>` prefixes the multi-hop synthesizers add
  to each context

The second was found only because a counter reported unmapped contexts: 8 of
11 unresolved on a trial run, leaving `reference_chunk_ids` empty on **every**
multi-hop row. Downstream that reads as "retrieval found nothing" on every run,
for ever, with no error.

A reconstruction that depends on text staying byte-identical through a
generation pipeline is not a reconstruction, it is a bet.

---

## What `app.ragas.io` is

Verified from `sdk.py` and `testset_schema.py:139`:

- a **hosted service**, not part of the package — no dashboard code ships in
  the wheel
- requires `RAGAS_APP_TOKEN`; a 403 raises `AUTHENTICATION_ERROR`
- `Testset.upload()` POSTs `samples_original` to
  `api.ragas.io/api/v1/alignment/testset` — the full test set, including
  questions, gold answers and retrieved contexts
- sends `x-ragas-lib-user-uuid`, a persistent installation id
- dashboards at `app.ragas.io/dashboard/alignment/{testset,evaluation}/{run_id}`

Separately, `_analytics.py` does usage telemetry with an opt-out
(`RAGAS_DO_NOT_TRACK=true`). That is distinct from upload and does not send
corpus data.

**Not verified: whether it is paywalled.** The framework is free and open
source; the dashboard's pricing tier was not confirmed.
[Issue #1954](https://github.com/explodinggradients/ragas/issues/1954) asks
whether the dashboard can be open-sourced and self-hosted — it is closed and
labelled "answered", but the reply was not retrievable. **Read that thread
before writing any of this**; it is the maintainers stating their position on
exactly this question.

The practical position today: viewing your test set means sending your corpus
to a third-party host, authenticated, or building it yourself.

---

## Proposed contribution

Four PRs, **zero new required dependencies**, ordered so the valuable parts
land even if the rest is declined.

### PR 1 — preserve provenance

Optional fields on `TestsetSample`: `node_ids: list[UUID]`, `persona`,
`style`, `length`. Populated at the one loop in `generate.py`.

~10 lines. No dependencies. **Independently valuable and not a visualisation
feature** — it makes a generated test set auditable, which is a correctness
property. It also enriches the payload the hosted app already receives.

### PR 2 — `KnowledgeGraph.diagnostics() -> dict`

Pure computation, no rendering: edge count per relationship type, edge density
as a fraction of possible pairs, cluster count per synthesizer, degree
distribution, and orphan nodes (missing `entities` or `summary_embedding`).

**Arguably worth more than any picture.** Those four numbers would have caught
three of this project's four failures before a cent was spent: 94.7% density,
zero `entities_overlap` edges, one orphan node.

### PR 3 — `KnowledgeGraph.to_html(path)`

One self-contained file. Data inlined as JSON, layout computed in Python,
rendered as inline SVG with vanilla JS for pan/zoom, selection and a
similarity-threshold slider. No CDN, no build step, no new packages. Mirrors
the existing `to_csv` / `to_jsonl` shape.

### PR 4 — `Testset.to_html(path)`

Joined to the graph through PR 1: each question with its persona, style,
synthesizer and a link back to its seed nodes.

---

## Why this shape

| alternative | why not |
|---|---|
| `pyvis` | pulls `jinja2` **and** `networkx` |
| `networkx` | a whole graph library for layout alone |
| `plotly` | ~30 MB, plus a CDN or bundled JS blob |
| `matplotlib` | static output cannot do the interaction that makes a 5,000-node graph legible, and is still a heavy new dependency |
| `_repr_html_` | idiomatic elsewhere, but ragas has zero IPython coupling — a new convention, and useless outside a notebook |
| a `ragas viz` CLI | no entry points exist today; packaging surface for little gain |
| upload to `app.ragas.io` | already exists, but needs a token and sends the corpus off-machine |

**Layout without new dependencies.** Fruchterman-Reingold vectorised in numpy
— already core — handles the realistic 400–5,000 node range in well under a
second. Above ~10k nodes, degrade to a deterministic layout keyed on a node
property rather than reaching for scipy.

**Testing.** Rendering is a pure function `(graph) -> str`: assert the payload
is embedded, that counts match the graph, and that **no external URL appears in
the output** — which makes the zero-dependency promise enforceable rather than
aspirational. No browser needed.

---

## Risks

- **PRs 3–4 overlap the hosted app.** A local, offline viewer is "look at your
  test set without an account". Propose as an issue first — not because of the
  money, but because the maintainers may already have a plan, and #1954
  suggests they have been asked before.
- **PRs 1–2 are correctness, not visualisation**, and should be argued that
  way. Provenance is destroyed by the library regardless of where the result is
  viewed.
- **Layout quality is a rabbit hole.** A force-directed layout in numpy is
  ~30 lines and adequate; making it *good* is not, and is not what the
  contribution is for. The diagnostics carry the value.

---

## What this repo does instead

`eval/testset.py` reconstructs provenance externally and defends the
reconstruction:

- feeds ragas our own chunks as nodes, one per row of `chunks`
- passes `transforms=` explicitly so no splitter can re-cut them
- strips the multi-hop marker before an exact dictionary lookup (`HOP_MARKER`)
- **counts unmapped contexts and reports them**, which is the only reason the
  marker bug was found

`assert_clusters` covers the same ground as PR 2's diagnostics for the three
synthesizers, by failing loudly when one would be silently dropped.
