import { useEffect, useState } from 'react'
import { listDocuments } from '../api'
import PdfPreview from '../components/PdfPreview'

const PAGE_SIZE = 20

export default function Documents() {
  const [docs, setDocs] = useState([])
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    listDocuments({ offset, limit: PAGE_SIZE })
      .then(setDocs)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [offset])

  const hasMore = docs.length === PAGE_SIZE

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Documents</h1>
        <p className="text-[13px] text-muted mt-0.5">All papers registered in the database.</p>
      </div>

      {error && <p className="text-[13px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5">{error}</p>}

      {loading ? (
        <p className="text-[13px] text-faint">Loading…</p>
      ) : docs.length === 0 && offset === 0 ? (
        <p className="text-[13px] text-faint">No documents yet. Use Fetch to add papers.</p>
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
                      <div className="font-medium truncate max-w-md">{doc.title}</div>
                      {authorsLabel && <div className="text-[11.5px] text-faint truncate max-w-md">{authorsLabel}</div>}
                    </td>
                    <td className="px-4 py-2.5 text-muted">{doc.venue ?? '—'}</td>
                    <td className="px-4 py-2.5 text-muted">{doc.year ?? '—'}</td>
                    <td className="px-4 py-2.5 text-right">
                      <PdfPreview docId={doc.id} title={doc.title} />
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
