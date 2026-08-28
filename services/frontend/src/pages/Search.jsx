import * as Tabs from '@radix-ui/react-tabs'
import { useState } from 'react'
import { search, searchKeyword } from '../api'
import PdfPreview from '../components/PdfPreview'

function ResultCard({ r }) {
  const authors = r.authors?.slice(0, 3).join(', ')
  const authorsLabel = r.authors?.length > 3 ? `${authors} et al.` : authors

  return (
    <article className="bg-surface border border-border rounded-xl p-4 space-y-2.5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold leading-snug">{r.title}</h3>
          <p className="text-xs text-muted mt-0.5">
            {authorsLabel}{r.year ? ` · ${r.year}` : ''}
          </p>
        </div>
        <PdfPreview docId={r.doc_id} title={r.title} />
      </div>
      <blockquote className="text-[13px] bg-canvas border-l-2 border-border rounded-r-lg px-3 py-2 leading-relaxed">
        {r.text}
      </blockquote>
      <div className="flex gap-3 text-[11px] text-faint">
        {r.page_num != null && <span>Page {r.page_num}</span>}
        <span className="ml-auto tabular-nums">score {r.score.toFixed(3)}</span>
      </div>
    </article>
  )
}

export default function Search() {
  const [mode, setMode] = useState('semantic')
  const [query, setQuery] = useState('')
  const [rerank, setRerank] = useState(false)
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSearch(e) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data =
        mode === 'semantic'
          ? await search(query, { topK, rerank, rerankTopK: 5 })
          : await searchKeyword(query, { topK })
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Search</h1>
        <p className="text-[13px] text-muted mt-0.5">
          {mode === 'semantic'
            ? 'Semantic search over indexed paper chunks - meaning, not exact words.'
            : 'Keyword search - literal word matches, ranked by relevance.'}
        </p>
      </div>

      <Tabs.Root value={mode} onValueChange={setMode}>
        <Tabs.List className="inline-flex rounded-lg border border-border bg-surface p-0.5 mb-4">
          <Tabs.Trigger
            value="semantic"
            className="px-3.5 py-1.5 rounded-md text-[13px] font-medium text-muted data-[state=active]:bg-canvas data-[state=active]:text-ink"
          >
            Semantic
          </Tabs.Trigger>
          <Tabs.Trigger
            value="keyword"
            className="px-3.5 py-1.5 rounded-md text-[13px] font-medium text-muted data-[state=active]:bg-canvas data-[state=active]:text-ink"
          >
            Keyword
          </Tabs.Trigger>
        </Tabs.List>
      </Tabs.Root>

      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={mode === 'semantic' ? 'Search research papers…' : 'Search exact words…'}
            className="flex-1 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm focus:outline-none focus:ring-1 focus:ring-ink/20"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-xl bg-accent text-accent-ink text-sm font-medium px-5 disabled:opacity-40"
          >
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>
        <div className="flex items-center gap-5 text-[13px] text-muted">
          {mode === 'semantic' && (
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
              Rerank results
            </label>
          )}
          <label className="flex items-center gap-2">
            Top
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="rounded-md border border-border bg-surface px-2 py-1 text-[13px]"
            >
              {[5, 10, 20, 50].map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        </div>
      </form>

      {error && <p className="text-[13px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5">{error}</p>}

      {results && (
        <div className="space-y-3">
          <p className="text-[13px] text-muted">
            {results.results.length} results{results.reranked ? ' · reranked' : ''} ·{' '}
            <code>{results.model}</code>
          </p>
          {results.results.length === 0 ? (
            <p className="text-[13px] text-faint">No results found.</p>
          ) : (
            results.results.map((r) => <ResultCard key={r.chunk_id} r={r} />)
          )}
        </div>
      )}
    </div>
  )
}
