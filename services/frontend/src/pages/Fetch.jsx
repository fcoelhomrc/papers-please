import { useState } from 'react'
import { fetchPapers } from '../api'

export default function Fetch() {
  const [form, setForm] = useState({ query: '', venue: '', year: '', maxPapers: 500 })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const set = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }))

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const data = await fetchPapers({
        query: form.query,
        venue: form.venue,
        year: form.year,
        maxPapers: Number(form.maxPapers),
      })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Fetch Papers</h1>
        <p className="text-[13px] text-muted mt-0.5">
          Query Semantic Scholar and register new papers for download and indexing. Papers
          already in the library are skipped automatically.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-surface border border-border rounded-xl p-5 space-y-4">
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted">Query</label>
          <input
            type="text"
            value={form.query}
            onChange={set('query')}
            placeholder="e.g. attention transformers"
            className="w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ink/20"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted">Venue</label>
            <input
              type="text"
              value={form.venue}
              onChange={set('venue')}
              placeholder="e.g. NeurIPS"
              className="w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ink/20"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted">Year</label>
            <input
              type="text"
              value={form.year}
              onChange={set('year')}
              placeholder="e.g. 2023 or 2020-2024"
              className="w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ink/20"
            />
          </div>
        </div>
        <div className="space-y-1">
          <label className="text-xs font-medium text-muted">Max papers</label>
          <input
            type="number"
            value={form.maxPapers}
            onChange={set('maxPapers')}
            min={1}
            max={5000}
            className="w-full rounded-lg border border-border bg-canvas px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ink/20"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-accent text-accent-ink text-sm font-medium py-2.5 disabled:opacity-40"
        >
          {loading ? 'Fetching…' : 'Fetch'}
        </button>
      </form>

      {error && <p className="text-[13px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5">{error}</p>}

      {result && (
        <p className="text-[13px] text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-2.5">
          Added <strong>{result.fetched}</strong> new papers (duplicates already in the library were
          skipped). The pipeline will download and index them shortly.
        </p>
      )}
    </div>
  )
}
