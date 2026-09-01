import { ArrowLeft, ExternalLink, FileText, SearchIcon } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Passage from '../components/Passage.jsx'
import PdfPreview from '../components/PdfPreview.jsx'
import {
  Badge,
  Card,
  EmptyState,
  ErrorState,
  Input,
  PageHeader,
  Skeleton,
} from '../components/ui.jsx'
import { useDocument, useDocumentChunks } from '../hooks/queries'

/* A paper's own page.
 *
 * Before this the only way to see a paper was the preview modal on a search
 * result: nothing was linkable, nothing was shareable, and the chunks the
 * index actually holds for a paper were observable only by searching for a
 * phrase you already knew was in it.
 */

function authorLine(authors, year, venue) {
  const shown = authors?.slice(0, 6).join(', ')
  const label = authors?.length > 6 ? `${shown} et al.` : shown
  return [label, venue, year].filter(Boolean).join(' · ')
}

// Semantic Scholar ids arrive as bare 40-char hashes or as "arXiv:2401.00001"
// style prefixed ids. Only the ones we can turn into a real URL get a link;
// the rest render as plain text rather than a link that 404s.
function sourceUrl(sourceId) {
  if (!sourceId) return null
  const arxiv = sourceId.match(/^(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)$/i)
  if (arxiv) return `https://arxiv.org/abs/${arxiv[1]}`
  if (/^10\.\d{4,9}\//.test(sourceId)) return `https://doi.org/${sourceId}`
  if (/^[0-9a-f]{40}$/i.test(sourceId)) return `https://www.semanticscholar.org/paper/${sourceId}`
  return null
}

export default function Document() {
  const { docId } = useParams()
  const [filter, setFilter] = useState('')
  const doc = useDocument(docId)
  const chunks = useDocumentChunks(docId)

  if (doc.error) return <ErrorState error={doc.error} />
  if (doc.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-7 w-3/4" />
        <Skeleton className="h-4 w-1/2" />
        <Skeleton className="h-32 w-full" />
      </div>
    )
  }

  const paper = doc.data
  const external = sourceUrl(paper.source_id)

  // Filtering client-side: the chunks are already loaded, so a round trip
  // would make an instant interaction slower for no additional capability.
  // Substring, not retrieval - "search within this paper" here means
  // "find where this word appears", which is what a reader looking at one
  // paper actually wants.
  const needle = filter.trim().toLowerCase()
  const rows = (chunks.data || []).filter(
    (c) => !needle || c.text.toLowerCase().includes(needle),
  )

  return (
    <div className="space-y-6">
      <Link
        to="/documents"
        className="inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
      >
        <ArrowLeft size={14} /> All papers
      </Link>

      <PageHeader
        title={paper.title}
        description={authorLine(paper.authors, paper.year, paper.venue)}
        actions={
          paper.has_pdf ? (
            <PdfPreview docId={paper.id} title={paper.title} label="Open PDF" size="md" />
          ) : null
        }
      />

      <div className="flex flex-wrap items-center gap-2">
        {paper.processed ? (
          <Badge tone="accent">searchable</Badge>
        ) : paper.has_pdf ? (
          <Badge>PDF only — not yet indexed</Badge>
        ) : (
          <Badge>metadata only</Badge>
        )}
        {external && (
          <a
            href={external}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-muted transition-colors hover:text-ink"
          >
            <ExternalLink size={12} /> source
          </a>
        )}
      </div>

      {paper.abstract && (
        <Card className="p-4">
          <h2 className="mb-2 text-sm font-semibold">Abstract</h2>
          <p className="text-sm leading-relaxed text-muted">{paper.abstract}</p>
        </Card>
      )}

      <section className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm font-semibold">
            Indexed passages
            {chunks.data && <span className="ml-2 text-muted">{chunks.data.length}</span>}
          </h2>
          {chunks.data?.length > 0 && (
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Find in this paper…"
              className="ml-auto w-full sm:w-64"
            />
          )}
        </div>

        {chunks.isLoading ? (
          <div className="space-y-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : !chunks.data?.length ? (
          <EmptyState
            icon={FileText}
            title="Not indexed yet"
            description="This paper has no chunks — it hasn't been through OCR and embedding, so it won't appear in search results."
          />
        ) : !rows.length ? (
          <EmptyState
            icon={SearchIcon}
            title="No matches"
            description={`Nothing in this paper contains “${filter}”.`}
          />
        ) : (
          <div className="space-y-2">
            {rows.map((c) => (
              <Card key={c.chunk_id} className="p-3">
                <div className="mb-1.5 flex items-center gap-2 text-2xs text-faint">
                  <span className="tabular-nums">#{c.chunk_index}</span>
                  {c.page_num != null && <span>page {c.page_num}</span>}
                </div>
                <Passage text={c.text} />
              </Card>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
