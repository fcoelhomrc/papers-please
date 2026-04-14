import { useState } from 'react'
import { pdfUrl, search } from '../api'

function ResultCard({ result }) {
  const authors = result.authors?.slice(0, 3).join(', ')
  const authorsLabel = result.authors?.length > 3 ? `${authors} et al.` : authors

  return (
    <article className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="font-semibold text-gray-900 leading-snug">{result.title}</h3>
          <p className="text-sm text-gray-500 mt-0.5">
            {authorsLabel}
            {result.year ? ` · ${result.year}` : ''}
          </p>
        </div>
        <a
          href={pdfUrl(result.doc_id)}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 text-xs font-medium text-indigo-600 hover:text-indigo-800 border border-indigo-200 hover:border-indigo-400 rounded-md px-2.5 py-1 transition-colors"
        >
          PDF
        </a>
      </div>

      <blockquote className="text-sm text-gray-700 bg-gray-50 rounded-lg px-4 py-3 border-l-4 border-indigo-200 leading-relaxed line-clamp-4">
        {result.text}
      </blockquote>

      <div className="flex items-center gap-3 text-xs text-gray-400">
        {result.page_num != null && <span>Page {result.page_num}</span>}
        <span className="ml-auto tabular-nums">score {result.score.toFixed(3)}</span>
      </div>
    </article>
  )
}

export default function Search() {
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
      const data = await search(query, { topK, rerank, rerankTopK: 5 })
      setResults(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search research papers…"
            className="flex-1 rounded-xl border border-gray-300 bg-white px-4 py-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium px-5 py-3 transition-colors"
          >
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>

        <div className="flex items-center gap-6 text-sm text-gray-600">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={rerank}
              onChange={(e) => setRerank(e.target.checked)}
              className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
            Rerank results
          </label>
          <label className="flex items-center gap-2">
            Top
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="rounded-md border border-gray-300 bg-white px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {[5, 10, 20, 50].map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </label>
        </div>
      </form>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
          {error}
        </p>
      )}

      {results && (
        <div className="space-y-4">
          <p className="text-sm text-gray-500">
            {results.results.length} results
            {results.reranked ? ' · reranked' : ''}
            {' · '}
            <span className="font-mono">{results.model}</span>
          </p>
          {results.results.length === 0 ? (
            <p className="text-gray-400 text-sm">No results found.</p>
          ) : (
            results.results.map((r) => <ResultCard key={r.chunk_id} result={r} />)
          )}
        </div>
      )}
    </div>
  )
}
