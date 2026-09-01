import { FileText, SearchIcon, SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import Passage from '../components/Passage.jsx'
import PdfPreview from '../components/PdfPreview.jsx'
import {
  Badge,
  Button,
  Card,
  Checkbox,
  EmptyState,
  ErrorState,
  Input,
  PageHeader,
  SegmentedControl,
  Select,
  Skeleton,
} from '../components/ui.jsx'
import { useSearch } from '../hooks/queries'

const MODE_HELP = {
  semantic: 'Semantic search over indexed paper chunks — matches meaning, not exact words.',
  keyword: 'Keyword search — literal word matches, ranked by Postgres full-text relevance.',
  hybrid: 'Both retrievers, fused by reciprocal rank — chunks that rank well in each win.',
}

const MODE_PLACEHOLDER = {
  semantic: 'Search research papers…',
  keyword: 'Search exact words…',
  hybrid: 'Search research papers…',
}

function authorLine(authors, year) {
  const shown = authors?.slice(0, 3).join(', ')
  const label = authors?.length > 3 ? `${shown} et al.` : shown
  return [label, year].filter(Boolean).join(' · ')
}

// Which retriever(s) surfaced this chunk. Agreement between two independent
// retrievers is a different kind of confidence from one being very sure, and
// the score alone can't express that - the backend computed it during fusion
// and used to throw it away.
const SOURCE_LABEL = { semantic: 'meaning', keyword: 'exact words' }

function ResultCard({ r }) {
  return (
    <Card className="animate-slide-up p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-semibold leading-snug">
            <Link to={`/documents/${r.doc_id}`} className="hover:text-accent">
              {r.title}
            </Link>
          </h3>
          <p className="mt-0.5 text-xs text-muted">{authorLine(r.authors, r.year)}</p>
        </div>
        {/* The page the passage came from, so the preview opens there
            instead of at page 1. */}
        <PdfPreview docId={r.doc_id} title={r.title} page={r.page_num} />
      </div>
      <Passage text={r.text} className="mt-3" />
      <div className="mt-2.5 flex flex-wrap items-center gap-2 text-2xs text-faint">
        {r.page_num != null && <span>Page {r.page_num}</span>}
        {r.sources?.map((s) => (
          <Badge key={s} tone={r.sources.length > 1 ? 'accent' : 'neutral'}>
            {SOURCE_LABEL[s] || s}
          </Badge>
        ))}
        <span className="ml-auto tabular-nums">score {r.score.toFixed(3)}</span>
      </div>
    </Card>
  )
}

function ResultSkeleton() {
  return (
    <Card className="space-y-3 p-4">
      <Skeleton className="h-4 w-2/3" />
      <Skeleton className="h-3 w-1/3" />
      <Skeleton className="h-16 w-full" />
    </Card>
  )
}

export default function Search() {
  const [mode, setMode] = useState('semantic')
  const [draft, setDraft] = useState('')
  const [showOptions, setShowOptions] = useState(false)
  const [rerank, setRerank] = useState(false)
  const [topK, setTopK] = useState(10)

  // `submitted` is what the query key reads - typing must not fire a request
  // per keystroke, and re-submitting an identical query serves from cache.
  const [submitted, setSubmitted] = useState('')

  const { data, error, isFetching } = useSearch({
    mode,
    query: submitted,
    topK,
    rerank,
    enabled: Boolean(submitted),
  })

  return (
    <div className="space-y-6">
      <PageHeader
        title="Search"
        description={MODE_HELP[mode]}
      />

      <form
        onSubmit={(e) => {
          e.preventDefault()
          setSubmitted(draft)
        }}
        className="space-y-3"
      >
        <div className="flex flex-col gap-2 sm:flex-row">
          <SegmentedControl
            value={mode}
            onChange={setMode}
            options={[
              { value: 'semantic', label: 'Semantic' },
              { value: 'keyword', label: 'Keyword' },
              { value: 'hybrid', label: 'Hybrid' },
            ]}
          />
          <div className="flex flex-1 gap-2">
            <Input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={MODE_PLACEHOLDER[mode]}
              className="flex-1"
            />
            <Button type="submit" variant="primary" loading={isFetching} disabled={!draft.trim()}>
              Search
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              onClick={() => setShowOptions((v) => !v)}
              aria-label="Search options"
              aria-expanded={showOptions}
            >
              <SlidersHorizontal size={15} />
            </Button>
          </div>
        </div>

        {showOptions && (
          <div className="flex flex-wrap items-center gap-4 rounded-lg border border-border bg-surface px-3.5 py-2.5 animate-fade-in">
            <Checkbox
              label="Rerank results"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
            />
            <label className="flex items-center gap-2 text-sm text-muted">
              Top
              <Select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
                {[5, 10, 20, 50].map((k) => (
                  <option key={k} value={k}>{k}</option>
                ))}
              </Select>
            </label>
          </div>
        )}
      </form>

      <ErrorState error={error} />

      {isFetching && !data ? (
        <div className="space-y-3">
          {[0, 1, 2].map((i) => <ResultSkeleton key={i} />)}
        </div>
      ) : !submitted ? (
        <EmptyState
          icon={SearchIcon}
          title="Search the library"
          description="Query indexed paper chunks by meaning or by exact keywords."
        />
      ) : data?.results.length === 0 ? (
        <EmptyState
          icon={FileText}
          title="No results"
          description="Nothing matched that query. Try different wording, or fetch more papers."
        />
      ) : data ? (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-muted">
            <span>{data.results.length} results</span>
            <Badge tone="accent">{data.mode}</Badge>
            {data.reranked && <Badge tone="accent">reranked</Badge>}
            <Badge>{data.model}</Badge>
          </div>
          {data.results.map((r) => <ResultCard key={r.chunk_id} r={r} />)}
        </div>
      ) : null}
    </div>
  )
}
