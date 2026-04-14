import { useEffect, useState } from 'react'
import { listDocuments, pdfUrl } from '../api'

const PAGE_SIZE = 20

export default function Documents() {
  const [docs, setDocs] = useState([])
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [hasMore, setHasMore] = useState(true)

  useEffect(() => {
    setLoading(true)
    setError(null)
    listDocuments({ offset, limit: PAGE_SIZE })
      .then((data) => {
        setDocs(data)
        setHasMore(data.length === PAGE_SIZE)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [offset])

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-gray-900">Documents</h2>
        <p className="text-sm text-gray-500 mt-1">All papers registered in the database.</p>
      </div>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-3">
          {error}
        </p>
      )}

      {loading ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="text-sm text-gray-400">No documents yet. Use Fetch to add papers.</p>
      ) : (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="text-left px-4 py-3 font-medium text-gray-600">Title</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600 hidden md:table-cell">Venue</th>
                <th className="text-left px-4 py-3 font-medium text-gray-600 hidden sm:table-cell">Year</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {docs.map((doc) => (
                <tr key={doc.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3">
                    <p className="font-medium text-gray-900 line-clamp-1">{doc.title}</p>
                    {doc.authors?.length > 0 && (
                      <p className="text-xs text-gray-400 mt-0.5 line-clamp-1">
                        {doc.authors.slice(0, 3).join(', ')}
                        {doc.authors.length > 3 ? ' et al.' : ''}
                      </p>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-500 hidden md:table-cell">{doc.venue ?? '—'}</td>
                  <td className="px-4 py-3 text-gray-500 hidden sm:table-cell">{doc.year ?? '—'}</td>
                  <td className="px-4 py-3 text-right">
                    <a
                      href={pdfUrl(doc.id)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs font-medium text-indigo-600 hover:text-indigo-800 border border-indigo-200 hover:border-indigo-400 rounded-md px-2.5 py-1 transition-colors"
                    >
                      PDF
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!loading && (offset > 0 || hasMore) && (
        <div className="flex justify-between items-center text-sm">
          <button
            onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
            disabled={offset === 0}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-gray-600 hover:bg-gray-100 disabled:opacity-40 transition-colors"
          >
            Previous
          </button>
          <span className="text-gray-400">
            {offset + 1}–{offset + docs.length}
          </span>
          <button
            onClick={() => setOffset((o) => o + PAGE_SIZE)}
            disabled={!hasMore}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-gray-600 hover:bg-gray-100 disabled:opacity-40 transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
