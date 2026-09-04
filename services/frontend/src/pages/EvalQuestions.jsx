import clsx from 'clsx'
import { Check, Pencil, Undo2, X } from 'lucide-react'
import { useState } from 'react'
import {
  Badge,
  Button,
  Callout,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  SegmentedControl,
  Skeleton,
  Textarea,
} from '../components/ui.jsx'
import { useEvalQuestions, useReviewQuestion } from '../hooks/queries'

/* Curating the generated test set: 131 in, ~100 out.
 *
 * Ragas' own docs expect 20-30% attrition, and there are two failure modes
 * worth cutting for. A question that *names its own answer* is retrieved by
 * BM25 on the entity alone, so every configuration scores identically and a
 * real difference between them disappears - insidious, because the question
 * reads perfectly well. A question with a floating "these approaches" means
 * nothing without the chunks it came from, and no user would ask it.
 *
 * Misspellings and poor grammar are NOT defects: ragas deliberately varies
 * QueryStyle across perfect/poor grammar, misspelled and web-search-like, and
 * that spread is the robustness half of the test set. */

const FILTERS = [
  { value: 'undecided', label: 'To review' },
  { value: 'keep', label: 'Kept' },
  { value: 'drop', label: 'Dropped' },
  { value: 'all', label: 'All' },
]

const SYNTH_LABEL = {
  single_hop_specifc_query_synthesizer: 'single-hop',
  single_hop_specific_query_synthesizer: 'single-hop',
  multi_hop_specific_query_synthesizer: 'multi-hop specific',
  multi_hop_abstract_query_synthesizer: 'multi-hop abstract',
}

function Counters({ summary, target }) {
  const topics = Object.entries(summary.kept_by_topic).sort()
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Badge tone={summary.kept === target ? 'success' : 'neutral'}>
          {summary.kept}/{target} kept
        </Badge>
        <Badge>{summary.undecided} to review</Badge>
        <Badge>{summary.dropped} dropped</Badge>
      </div>
      {topics.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {topics.map(([topic, n]) => (
            <Badge key={topic} tone="accent">
              {topic} {n}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

function QuestionCard({ q, onPatch, pending }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState({ question: q.question, reference: q.reference })
  const kept = q.decision === 'keep'
  const dropped = q.decision === 'drop'

  return (
    <Card className={clsx('p-4', kept && 'border-success/40', dropped && 'opacity-50')}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge>{SYNTH_LABEL[q.synthesizer] ?? q.synthesizer}</Badge>
          {q.topics.map((t) => (
            <Badge key={t} tone="accent">
              {t}
            </Badge>
          ))}
          {q.edited && <Badge tone="warn">edited</Badge>}
        </div>
        {kept && <Badge tone="success">kept</Badge>}
        {dropped && <Badge>dropped</Badge>}
      </div>

      {editing ? (
        <div className="mt-3 space-y-3">
          <div>
            <p className="mb-1 text-2xs font-medium uppercase tracking-wide text-faint">
              Question
            </p>
            <Textarea
              rows={2}
              value={draft.question}
              onChange={(e) => setDraft((d) => ({ ...d, question: e.target.value }))}
            />
          </div>
          <div>
            <p className="mb-1 text-2xs font-medium uppercase tracking-wide text-faint">
              Reference answer
            </p>
            <Textarea
              rows={3}
              value={draft.reference}
              onChange={(e) => setDraft((d) => ({ ...d, reference: e.target.value }))}
            />
          </div>
          {/* context_recall grades the pipeline's answer against `reference`.
              A reworded question with a stale gold answer marks correct
              answers wrong, silently, on every future run. */}
          <Callout tone="warn">
            Editing the question? Edit the reference to match — every metric
            that uses a gold answer grades against it.
          </Callout>
          <div className="flex gap-2">
            <Button
              size="sm"
              variant="primary"
              disabled={pending}
              onClick={() => {
                onPatch(q.id, draft)
                setEditing(false)
              }}
            >
              Save
            </Button>
            <Button size="sm" onClick={() => setEditing(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <>
          <p className="mt-3 text-sm font-medium leading-snug">{q.question}</p>
          <p className="mt-2 text-sm leading-relaxed text-muted">{q.reference}</p>
        </>
      )}

      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-muted hover:text-ink">
          {q.reference_contexts.length} seed chunk
          {q.reference_contexts.length === 1 ? '' : 's'} · chunk ids{' '}
          {q.reference_chunk_ids.join(', ') || '—'}
        </summary>
        <div className="mt-2 space-y-2">
          {q.reference_contexts.map((c, i) => (
            /* Wrapped, not horizontally scrolled: the reviewer is skimming a
               passage to judge whether it supports the answer, and a single
               long line hides everything past the fold. Newlines from the
               chunk are kept, since section breaks carry meaning. */
            <pre
              key={i}
              className="max-h-80 overflow-y-auto whitespace-pre-wrap break-words rounded-lg bg-inset p-3 text-2xs leading-relaxed text-muted"
            >
              {c}
            </pre>
          ))}
        </div>
      </details>

      <div className="mt-4 flex items-center gap-2">
        <Button
          size="sm"
          variant={kept ? 'secondary' : 'primary'}
          disabled={pending}
          onClick={() => onPatch(q.id, { decision: kept ? 'undecided' : 'keep' })}
        >
          {kept ? <Undo2 size={14} /> : <Check size={14} />}
          {kept ? 'Unkeep' : 'Keep'}
        </Button>
        <Button
          size="sm"
          disabled={pending}
          onClick={() => onPatch(q.id, { decision: dropped ? 'undecided' : 'drop' })}
        >
          {dropped ? <Undo2 size={14} /> : <X size={14} />}
          {dropped ? 'Undo' : 'Drop'}
        </Button>
        {!editing && (
          <Button size="sm" disabled={pending} onClick={() => setEditing(true)}>
            <Pencil size={14} />
            Edit
          </Button>
        )}
        <span className="ml-auto font-mono text-2xs text-faint">{q.id}</span>
      </div>
    </Card>
  )
}

export default function EvalQuestions() {
  const [filter, setFilter] = useState('undecided')
  const { data, isLoading, error } = useEvalQuestions()
  const review = useReviewQuestion()

  if (error) return <ErrorState error={error} />

  const all = data?.questions ?? []
  const shown = filter === 'all' ? all : all.filter((q) => q.decision === filter)

  return (
    <div className="space-y-6">
      <PageHeader
        title="Question review"
        description="Cut questions that name their own answer — BM25 retrieves those on the entity alone, so every retrieval config scores the same and real differences vanish. Cut abstract questions with a floating “these”, which mean nothing without their chunks. Misspellings and poor grammar are deliberate and worth keeping."
      />

      {data && <Counters summary={data.summary} target={data.target} />}

      <SegmentedControl value={filter} onChange={setFilter} options={FILTERS} />

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-40" />
          ))}
        </div>
      )}

      {!isLoading && shown.length === 0 && (
        <EmptyState
          title={filter === 'undecided' ? 'Nothing left to review' : 'Nothing here'}
          description={
            all.length === 0
              ? 'Run `uv run python -m eval.run_host testset generate` first.'
              : 'Switch filters to see the rest of the set.'
          }
        />
      )}

      <div className="space-y-3">
        {shown.map((q) => (
          <QuestionCard
            key={q.id}
            q={q}
            pending={review.isPending}
            onPatch={(qid, patch) => review.mutate({ qid, patch })}
          />
        ))}
      </div>
    </div>
  )
}
