import clsx from 'clsx'
import { AlertTriangle, ChevronDown, Loader2, SearchX } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { pdfUrl } from '../api'
import { citationIndex } from '../lib/citations.js'
import { Badge } from './ui.jsx'

/* What the agent found, and what it did to find it.
 *
 * The panel used to render the reply plus a row of inert tool-name chips: it
 * could say a search happened, but not what was searched for, what came
 * back, or whether it worked. Everything here comes from the `evidence` and
 * `trace` the API now returns (#31) — the data was already being computed
 * and thrown away one function call before the response was built.
 */

const TOOL_LABEL = {
  search_chunks: 'Searched the library',
  get_document: 'Looked up a paper',
  fetch_papers: 'Fetched new papers',
  get_status: 'Checked the pipeline',
}

function authorLine(authors, year) {
  const shown = authors?.slice(0, 2).join(', ')
  const label = authors?.length > 2 ? `${shown} et al.` : shown
  return [label, year].filter(Boolean).join(' · ')
}

export function EvidenceCards({ evidence }) {
  if (!evidence?.length) return null
  const index = citationIndex(evidence)

  return (
    <div className="mt-2.5 space-y-1.5 border-t border-border pt-2">
      <p className="text-2xs font-medium uppercase tracking-wide text-faint">Sources</p>
      {evidence.map((e) => (
        <a
          key={e.chunk_id ?? `${e.doc_id}-${e.page_num}`}
          // The anchor a citation superscript scrolls to. Several passages
          // from one paper share a number, so the id has to include the
          // chunk to stay unique — the citation targets the first.
          id={`evidence-${e.doc_id}`}
          href={pdfUrl(e.doc_id, { page: e.page_num })}
          target="_blank"
          rel="noreferrer"
          className="group flex gap-2 rounded-lg border border-border bg-canvas px-2.5 py-2 transition-colors hover:border-border-strong"
        >
          <span className="mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded bg-accent/15 text-2xs font-semibold tabular-nums text-accent">
            {index.get(e.doc_id)}
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-xs font-medium leading-snug group-hover:text-accent">
              {e.title || `Document ${e.doc_id}`}
            </span>
            <span className="mt-0.5 flex items-center gap-1.5 text-2xs text-faint">
              {authorLine(e.authors, e.year) && <span className="truncate">{authorLine(e.authors, e.year)}</span>}
              {e.page_num != null && <span className="shrink-0">p{e.page_num}</span>}
            </span>
          </span>
        </a>
      ))}
    </div>
  )
}

export function AbstentionNote() {
  return (
    <div className="mt-2.5 flex items-start gap-2 rounded-lg border border-border bg-inset px-2.5 py-2 text-2xs text-muted">
      <SearchX size={13} className="mt-px shrink-0 text-faint" />
      <span>
        Nothing in the library cleared the relevance threshold, so no sources are cited. This
        is retrieval declining, not failing.
      </span>
    </div>
  )
}

/* One line per tool call: what ran, with what, and what came back.
 *
 * Collapsed by default — the answer is what gets read, and a trace opened by
 * default would push it off screen. Auto-opened when something failed,
 * because that is the case where the trace is the actual message.
 */
export function ToolTrace({ trace }) {
  const failed = trace?.some((s) => !s.ok)
  const [open, setOpen] = useState(false)
  if (!trace?.length) return null

  const expanded = open || failed

  return (
    <div className="mt-2 border-t border-border pt-1.5">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 text-2xs text-faint transition-colors hover:text-muted"
        aria-expanded={expanded}
      >
        <ChevronDown
          size={11}
          className={clsx('transition-transform', expanded && 'rotate-180')}
        />
        {trace.length} step{trace.length !== 1 ? 's' : ''}
        {failed && (
          <Badge tone="danger" className="ml-1">
            failed
          </Badge>
        )}
      </button>

      {expanded && (
        <ul className="mt-1.5 space-y-1">
          {trace.map((step, i) => (
            <li key={i} className="flex gap-1.5 text-2xs leading-relaxed">
              <span className={clsx('mt-1 h-1 w-1 shrink-0 rounded-full', step.ok ? 'bg-faint' : 'bg-danger')} />
              <span className="min-w-0">
                <span className={step.ok ? 'text-muted' : 'text-danger'}>
                  {TOOL_LABEL[step.tool] || step.tool}
                </span>
                {step.args?.query && (
                  <span className="text-faint"> — “{step.args.query}”</span>
                )}
                {step.args?.doc_id != null && (
                  <span className="text-faint"> — doc {step.args.doc_id}</span>
                )}
                {step.summary && (
                  <span className="ml-1 text-faint">({step.summary})</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* What the agent is doing right now, from the SSE stream.
 *
 * Non-streaming, the panel showed a bouncing ellipsis for however long the
 * turn took and then everything landed at once — the tool calls were only
 * visible after they no longer mattered.
 */
export function LiveSteps({ steps }) {
  const last = steps?.[steps.length - 1]

  return (
    <div className="flex justify-start">
      <div className="flex max-w-[92%] items-center gap-2 rounded-2xl rounded-bl-md border border-border bg-canvas px-3.5 py-2 text-xs text-muted">
        <Loader2 size={12} className="shrink-0 animate-spin text-accent" />
        {last?.kind === 'tool_call' ? (
          <span className="min-w-0 truncate">
            {TOOL_LABEL[last.tool] || last.tool}
            {last.args?.query && <span className="text-faint"> — “{last.args.query}”</span>}
          </span>
        ) : (
          <span>Thinking…</span>
        )}
      </div>
    </div>
  )
}

export function StreamError({ message }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-danger/25 bg-danger-soft px-3 py-2 text-xs text-danger">
      <AlertTriangle size={13} className="mt-px shrink-0" />
      <span className="whitespace-pre-wrap">{message}</span>
    </div>
  )
}

export function DocumentLink({ docId, children }) {
  return (
    <Link to={`/documents/${docId}`} className="text-accent hover:underline">
      {children}
    </Link>
  )
}
