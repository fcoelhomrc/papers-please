import { CheckCircle2, FileStack } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import PdfPreview from '../components/PdfPreview.jsx'
import {
  Button,
  Card,
  Checkbox,
  EmptyState,
  ErrorState,
  Input,
  PageHeader,
  Select,
  Skeleton,
} from '../components/ui.jsx'
import { useDocuments } from '../hooks/queries'

const PAGE_SIZE = 20

export default function Documents() {
  const [offset, setOffset] = useState(0)
  const [filters, setFilters] = useState({
    q: '',
    onlyAvailable: false,
    onlyProcessed: false,
    sort: 'newest',
  })

  // Any filter change invalidates the current page position - staying on
  // page 3 of a different result set is never what you want.
  function setFilter(key, value) {
    setOffset(0)
    setFilters((f) => ({ ...f, [key]: value }))
  }

  const { data: docs, error, isLoading, isPlaceholderData } = useDocuments({
    offset,
    limit: PAGE_SIZE,
    ...filters,
  })

  const hasMore = docs?.length === PAGE_SIZE
  const filtersActive = filters.q || filters.onlyAvailable || filters.onlyProcessed

  return (
    <div className="space-y-6">
      <PageHeader title="Documents" description="All papers registered in the database." />

      {/* One row from sm up; on phones the filter input takes the full width
          and the toggles/sort share the row below it, rather than wrapping
          into three ragged lines. */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <Input
          value={filters.q}
          onChange={(e) => setFilter('q', e.target.value)}
          placeholder="Filter by title…"
          className="sm:min-w-[180px] sm:flex-1"
        />
        <div className="flex items-center gap-4">
          <Checkbox
            label="Has PDF"
            checked={filters.onlyAvailable}
            onChange={(e) => setFilter('onlyAvailable', e.target.checked)}
          />
          <Checkbox
            label="Searchable"
            checked={filters.onlyProcessed}
            onChange={(e) => setFilter('onlyProcessed', e.target.checked)}
          />
          <Select
            value={filters.sort}
            onChange={(e) => setFilter('sort', e.target.value)}
            className="ml-auto sm:ml-0"
          >
            <option value="newest">Newest</option>
            <option value="oldest">Oldest</option>
            <option value="title">Title</option>
            <option value="year">Year</option>
          </Select>
        </div>
      </div>

      <ErrorState error={error} />

      {isLoading ? (
        <Card className="divide-y divide-border">
          {Array.from({ length: 6 }, (_, i) => (
            <div key={i} className="flex items-center gap-4 p-4">
              <Skeleton className="h-4 flex-1" />
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-6 w-20" />
            </div>
          ))}
        </Card>
      ) : docs?.length === 0 ? (
        <EmptyState
          icon={FileStack}
          title={filtersActive ? 'No matching documents' : 'No documents yet'}
          description={
            filtersActive
              ? 'No documents match these filters.'
              : 'Use Fetch to add papers to the library.'
          }
        />
      ) : docs ? (
        <>
          {/* Table from sm up; the same rows as stacked cards on phones, where
              a four-column table would either overflow or truncate to noise. */}
          <Card
            className={isPlaceholderData ? 'overflow-hidden opacity-60 transition-opacity' : 'overflow-hidden'}
          >
            <table className="hidden w-full text-sm sm:table">
              <thead className="border-b border-border bg-inset">
                <tr>
                  <th className="px-4 py-2.5 text-left font-medium text-muted">Title</th>
                  <th className="px-4 py-2.5 text-left font-medium text-muted">Venue</th>
                  <th className="px-4 py-2.5 text-left font-medium text-muted">Year</th>
                  <th className="px-4 py-2.5" />
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {docs.map((doc) => (
                  <tr key={doc.id} className="transition-colors hover:bg-inset/60">
                    <td className="max-w-md px-4 py-2.5">
                      <div className="flex items-center gap-1.5">
                        {doc.processed && (
                          <CheckCircle2 size={13} className="shrink-0 text-success" />
                        )}
                        <Link
                          to={`/documents/${doc.id}`}
                          className="truncate font-medium transition-colors hover:text-accent"
                          title={doc.title}
                        >
                          {doc.title}
                        </Link>
                      </div>
                      <AuthorLine authors={doc.authors} />
                    </td>
                    <td className="px-4 py-2.5 text-muted">{doc.venue ?? '—'}</td>
                    <td className="px-4 py-2.5 tabular-nums text-muted">{doc.year ?? '—'}</td>
                    <td className="px-4 py-2.5 text-right">
                      {doc.has_pdf ? (
                        <PdfPreview docId={doc.id} title={doc.title} />
                      ) : (
                        <span className="text-2xs text-faint">no PDF yet</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="divide-y divide-border sm:hidden">
              {docs.map((doc) => (
                <div key={doc.id} className="space-y-2 p-4">
                  <div className="flex items-start gap-1.5">
                    {doc.processed && (
                      <CheckCircle2 size={13} className="mt-1 shrink-0 text-success" />
                    )}
                    <Link
                      to={`/documents/${doc.id}`}
                      className="text-sm font-medium leading-snug transition-colors hover:text-accent"
                    >
                      {doc.title}
                    </Link>
                  </div>
                  <AuthorLine authors={doc.authors} />
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-2xs text-muted">
                      {[doc.venue, doc.year].filter(Boolean).join(' · ') || '—'}
                    </span>
                    {doc.has_pdf ? (
                      <PdfPreview docId={doc.id} title={doc.title} />
                    ) : (
                      <span className="text-2xs text-faint">no PDF yet</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {(offset > 0 || hasMore) && (
            <div className="flex items-center justify-between text-sm">
              <Button
                size="sm"
                onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                disabled={offset === 0}
              >
                Previous
              </Button>
              <span className="tabular-nums text-faint">
                {offset + 1}–{offset + docs.length}
              </span>
              <Button size="sm" onClick={() => setOffset((o) => o + PAGE_SIZE)} disabled={!hasMore}>
                Next
              </Button>
            </div>
          )}
        </>
      ) : null}
    </div>
  )
}

function AuthorLine({ authors }) {
  if (!authors?.length) return null
  const shown = authors.slice(0, 3).join(', ')
  return (
    <div className="truncate text-2xs text-faint">
      {authors.length > 3 ? `${shown} et al.` : shown}
    </div>
  )
}
