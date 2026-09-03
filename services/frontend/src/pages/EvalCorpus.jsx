import clsx from 'clsx'
import { Check, Undo2, X } from 'lucide-react'
import { useState } from 'react'
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  SegmentedControl,
  Skeleton,
} from '../components/ui.jsx'
import { useEvalCandidates, useEvalDecision } from '../hooks/queries'

/* Curating the evaluation corpus: 40 staged candidates per topic, 20 kept.
 *
 * Taking the top 20 by citation would produce a list, not a corpus. What
 * makes retrieval hard is within-topic neighbours - competing methods for the
 * same problem, v1/v2 of an idea - so the reviewer keeps near-duplicates on
 * purpose and mixes short papers with long surveys.
 *
 * Every paper stays in the list after a decision. Kept ones because the
 * judgement is comparative - reviewing against a bare counter means working
 * blind about what is already in the topic - and rejected ones because a
 * decision you cannot see is a decision you cannot undo. */

function TopicTabs({ topics, value, onChange, target }) {
  return (
    <div className="space-y-3">
      <SegmentedControl
        value={value}
        onChange={onChange}
        options={topics.map((t) => ({ value: t.topic, label: t.topic }))}
      />
      <div className="flex flex-wrap gap-2">
        {topics.map((t) => (
          <Badge
            key={t.topic}
            tone={t.kept === target ? 'success' : t.kept > target ? 'warn' : 'neutral'}
          >
            {t.topic} {t.kept}/{target}
          </Badge>
        ))}
      </div>
    </div>
  )
}

function CandidateCard({ paper, onDecide, pending }) {
  const [expanded, setExpanded] = useState(false)
  const kept = paper.corpus === 'eval'
  const rejected = paper.corpus === 'main'
  const abstract = paper.abstract ?? ''
  // Long abstracts make the list unscannable, but the decision genuinely
  // needs the method sentence, which is rarely in the first line.
  const clipped = abstract.length > 320 && !expanded

  return (
    <Card
      className={clsx('p-4', kept && 'border-success/40', rejected && 'opacity-50')}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-medium leading-snug">{paper.title}</h3>
          <p className="mt-1 text-xs text-muted">
            {paper.year ?? 'n.d.'} · {paper.citation_count ?? 0} citations
            {paper.authors?.length ? ` · ${paper.authors[0]}${paper.authors.length > 1 ? ' et al.' : ''}` : ''}
          </p>
        </div>
        {kept && <Badge tone="success">kept</Badge>}
        {rejected && <Badge>rejected</Badge>}
      </div>

      {abstract && (
        <p className="mt-3 text-sm leading-relaxed text-muted">
          {clipped ? `${abstract.slice(0, 320)}…` : abstract}
          {abstract.length > 320 && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="ml-1 text-accent hover:underline"
            >
              {expanded ? 'less' : 'more'}
            </button>
          )}
        </p>
      )}

      <div className="mt-4 flex items-center gap-2">
        {rejected ? (
          <Button size="sm" disabled={pending} onClick={() => onDecide(paper.id, 'reset')}>
            <Undo2 size={14} />
            Undo
          </Button>
        ) : (
          <>
            <Button
              size="sm"
              variant={kept ? 'secondary' : 'primary'}
              disabled={pending}
              onClick={() => onDecide(paper.id, kept ? 'reset' : 'keep')}
            >
              {kept ? <Undo2 size={14} /> : <Check size={14} />}
              {kept ? 'Unkeep' : 'Keep'}
            </Button>
            <Button size="sm" disabled={pending} onClick={() => onDecide(paper.id, 'reject')}>
              <X size={14} />
              Reject
            </Button>
          </>
        )}
        <a
          href={paper.pdf_url}
          target="_blank"
          rel="noreferrer"
          className="ml-auto text-xs text-muted hover:text-ink hover:underline"
        >
          arXiv PDF
        </a>
      </div>
    </Card>
  )
}

export default function EvalCorpus() {
  const [topic, setTopic] = useState(null)
  const { data, isLoading, error } = useEvalCandidates(topic)
  const decision = useEvalDecision()

  if (error) return <ErrorState error={error} />

  const topics = data?.topics ?? []
  const target = data?.target_per_topic ?? 20
  // Default to the first topic once the summary arrives, rather than showing
  // all 213 candidates at once - the decision is per-topic.
  const active = topic ?? topics[0]?.topic ?? null
  const shown = (data?.candidates ?? []).filter((c) => !active || c.topic === active)
  const totalKept = topics.reduce((n, t) => n + t.kept, 0)

  return (
    <div className="space-y-6">
      <PageHeader
        title="Corpus curation"
        description="Keep 20 papers per topic from the staged candidates. Prefer competing methods on the same problem and a mix of paper lengths — within-topic neighbours are what make retrieval hard to measure."
        actions={
          <Badge tone={totalKept === target * topics.length ? 'success' : 'neutral'}>
            {totalKept}/{target * (topics.length || 1)} kept
          </Badge>
        }
      />

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      )}

      {!isLoading && topics.length > 0 && (
        <TopicTabs topics={topics} value={active} onChange={setTopic} target={target} />
      )}

      {!isLoading && shown.length === 0 && (
        <EmptyState
          title="Nothing staged"
          description="Run `uv run python -m eval.corpus stage` to fetch candidates."
        />
      )}

      <div className="space-y-3">
        {shown.map((paper) => (
          <CandidateCard
            key={paper.id}
            paper={paper}
            pending={decision.isPending}
            onDecide={(docId, d) => decision.mutate({ docId, decision: d })}
          />
        ))}
      </div>
    </div>
  )
}
