import { useState } from 'react'
import { fetchPapers } from '../api'

export default function Fetch() {
  const [form, setForm] = useState({ query: '', venue: '', year: '', maxPapers: 500 })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  function set(field) {
    return (e) => setForm((f) => ({ ...f, [field]: e.target.value }))
  }

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
        <h2 className="text-lg font-semibold text-gray-900">Fetch Papers</h2>
        <p className="text-sm text-gray-500 mt-1">
          Query Semantic Scholar and register new papers for download and indexing.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 space-y-4">
        <div className="space-y-1">
          <label className="text-sm font-medium text-gray-700">Query</label>
          <input
            type="text"
            value={form.query}
            onChange={set('query')}
            placeholder="e.g. attention transformers"
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700">Venue</label>
            <input
              type="text"
              value={form.venue}
              onChange={set('venue')}
              placeholder="e.g. NeurIPS"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          <div className="space-y-1">
            <label className="text-sm font-medium text-gray-700">Year</label>
            <input
              type="text"
              value={form.year}
              onChange={set('year')}
              placeholder="e.g. 2023 or 2020-2024"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-sm font-medium text-gray-700">Max papers</label>
          <input
            type="number"
            value={form.maxPapers}
            onChange={set('maxPapers')}
            min={1}
            max={5000}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium py-2.5 transition-colors"
        >
          {loading ? 'Fetching…' : 'Fetch'}
        </button>
      </form>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
          {error}
        </p>
      )}

      {result && (
        <p className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg px-4 py-3">
          Registered <strong>{result.fetched}</strong> papers. The worker will download and index them shortly.
        </p>
      )}
    </div>
  )
}
