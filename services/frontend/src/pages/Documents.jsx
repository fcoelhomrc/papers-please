import { CheckCircle2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { listDocuments } from '../api'
import PdfPreview from '../components/PdfPreview'

const PAGE_SIZE = 20

export default function Documents() {
  const [docs, setDocs] = useState([])
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const [q, setQ] = useState('')
  const [onlyAvailable, setOnlyAvailable] = useState(false)
  const [onlyProcessed, setOnlyProcessed] = useState(false)
  const [sort, setSort] = useState('newest')

  useEffect(() => {
    setLoading(true)
    setError(null)
    listDocuments({ offset, limit: PAGE_SIZE, q, onlyAvailable, onlyProcessed, sort })
      .then(setDocs)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [offset, q, onlyAvailable, onlyProcessed, sort])

  function resetAndSet(setter) {
    return (value) => {
      setOffset(0)
      setter(value)
    }
  }

  const hasMore = docs.length === PAGE_SIZE
  const filtersActive = q || onlyAvailable || onlyProcessed

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Documents</h1>
        <p className="text-[13px] text-muted mt-0.5">All papers registered in the database.</p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <input
          type="text"
          value={q}
          onChange={(e) => resetAndSet(setQ)(e.target.value)}
          placeholder="Filter by title…"
          className="flex-1 min-w-[180px] rounded-lg border border-border bg-surface px-3 py-1.5 text-[13px] focus:outline-none focus:ring-1 focus:ring-ink/20"
        />
        <label className="flex items-center gap-1.5 text-[13px] text-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={onlyAvailable}
            onChange={(e) => resetAndSet(setOnlyAvailable)(e.target.checked)}
          />
          Has PDF
        </label>
        <label className="flex items-center gap-1.5 text-[13px] text-muted cursor-pointer select-none">
          <input
            type="checkbox"
            checked={onlyProcessed}
            onChange={(e) => resetAndSet(setOnlyProcessed)(e.target.checked)}
          />
          Searchable
        </label>
        <select
          value={sort}
          onChange={(e) => resetAndSet(setSort)(e.target.value)}
          className="rounded-lg border border-border bg-surface px-2 py-1.5 text-[13px]"
        >
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
          <option value="title">Title</option>
          <option value="year">Year</option>
        </select>
      </div>

      {error && <p className="text-[13px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5">{error}</p>}

      {loading ? (
        <p className="text-[13px] text-faint">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="text-[13px] text-faint">
          {filtersActive ? 'No documents match these filters.' : 'No documents yet. Use Fetch to add papers.'}
        </p>
      ) : (
        <div className="bg-surface border border-border rounded-xl overflow-hidden">
          <table className="w-full text-[13px]">
            <thead className="bg-canvas border-b border-border">
              <tr>
                <th className="text-left px-4 py-2.5 font-medium text-muted">Title</th>
                <th className="text-left px-4 py-2.5 font-medium text-muted">Venue</th>
                <th className="text-left px-4 py-2.5 font-medium text-muted">Year</th>
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {docs.map((doc) => {
                const authors = doc.authors?.slice(0, 3).join(', ')
                const authorsLabel = doc.authors?.length > 3 ? `${authors} et al.` : authors
                return (
                  <tr key={doc.id}>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-1.5">
                        {doc.processed && (
                          <CheckCircle2 size={13} className="text-green-600 dark:text-green-400 shrink-0" />
                        )}
                        <span className="font-medium truncate max-w-md" title={doc.processed ? 'Searchable' : undefined}>
                          {doc.title}
                        </span>
                      </div>
                      {authorsLabel && <div className="text-[11.5px] text-faint truncate max-w-md">{authorsLabel}</div>}
                    </td>
                    <td className="px-4 py-2.5 text-muted">{doc.venue ?? '—'}</td>
                    <td className="px-4 py-2.5 text-muted">{doc.year ?? '—'}</td>
                    <td className="px-4 py-2.5 text-right">
                      {doc.has_pdf ? (
                        <PdfPreview docId={doc.id} title={doc.title} />
                      ) : (
                        <span className="text-[11.5px] text-faint">no PDF yet</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {!loading && (offset > 0 || hasMore) && (
        <div className="flex justify-between items-center text-[13px]">
          <button
            onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
            disabled={offset === 0}
            className="rounded-lg border border-border px-3 py-1.5 disabled:opacity-40"
          >
            Previous
          </button>
          <span className="text-faint">{offset + 1}–{offset + docs.length}</span>
          <button
            onClick={() => setOffset((o) => o + PAGE_SIZE)}
            disabled={!hasMore}
            className="rounded-lg border border-border px-3 py-1.5 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
