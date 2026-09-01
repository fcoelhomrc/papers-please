import clsx from 'clsx'
import { Inbox } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Badge, Card, EmptyState, ErrorState, PageHeader, Skeleton } from '../components/ui.jsx'
import WorkerStrip from '../components/WorkerStrip.jsx'
import { useQueue, useStatus, useWorkers } from '../hooks/queries'

const REFRESH_MS = 10_000

/* Aggregate counts answer "how much is left"; the list answers "what is it
 * doing right now", which the counters could not - a document stuck on its
 * third OCR attempt looked exactly like a busy pipeline.
 */
const STATUS = {
  dead: { label: 'gave up', tone: 'danger', hint: 'Chunking failed repeatedly — this PDF will not be retried' },
  // A download that gave up and an OCR failure are different problems with
  // different fixes: one means the URL is bad, the other means the PDF is.
  download_failed: {
    label: 'no download',
    tone: 'danger',
    hint: 'The PDF URL failed repeatedly (often a DOI link to a paywall) — not retried',
  },
  failed: { label: 'failed', tone: 'warn', hint: 'Chunking failed; it will be retried' },
  awaiting_download: { label: 'queued', tone: 'neutral', hint: 'Waiting for the download stage to pick it up' },
  downloading: { label: 'downloading', tone: 'neutral', hint: 'Download in progress or being retried' },
  pending: { label: 'chunking', tone: 'neutral', hint: 'Waiting for OCR and chunking' },
  chunked: { label: 'embedding', tone: 'neutral', hint: 'Chunked; waiting for embeddings' },
  embedded: { label: 'searchable', tone: 'accent', hint: 'Fully indexed' },
  metadata_only: { label: 'no PDF', tone: 'neutral', hint: 'No PDF URL — cannot be indexed' },
}

function Stat({ label, value, tone = 'default' }) {
  const toneClass = { default: 'text-ink', warn: 'text-warn', ok: 'text-success' }[tone]
  return (
    <Card className="p-4">
      <div className="text-2xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx('mt-1.5 text-3xl font-semibold tabular-nums', toneClass)}>{value}</div>
    </Card>
  )
}

function QueueRow({ item }) {
  const s = STATUS[item.status] || { label: item.status, tone: 'neutral', hint: '' }

  return (
    <div className="flex items-center gap-3 border-b border-border px-3 py-2.5 last:border-0">
      <div className="min-w-0 flex-1">
        <Link
          to={`/documents/${item.doc_id}`}
          className="block truncate text-sm transition-colors hover:text-accent"
          title={item.title}
        >
          {item.title}
        </Link>
        <div className="mt-0.5 flex items-center gap-2 text-2xs text-faint">
          {item.chunks > 0 && (
            <span className="tabular-nums">
              {item.embedded}/{item.chunks} chunks embedded
            </span>
          )}
          {/* Only shown once something has actually gone wrong - an attempt
              counter on every healthy row is noise. */}
          {item.attempts > 0 && (
            <span className="tabular-nums">
              {item.attempts} attempt{item.attempts !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>
      {/* Wrapped rather than passing title to Badge: Badge doesn't forward
          unknown props, so the tooltip would be silently dropped. */}
      <span title={s.hint} className="shrink-0">
        <Badge tone={s.tone}>{s.label}</Badge>
      </span>
    </div>
  )
}

export default function Queue() {
  const { data, error, isLoading, isFetching } = useStatus({ refetchInterval: REFRESH_MS })
  const queue = useQueue({ refetchInterval: REFRESH_MS })
  const workers = useWorkers({ refetchInterval: REFRESH_MS })

  const pendingChunk = data?.objects_by_status?.pending ?? 0
  const failedChunk = data?.objects_by_status?.failed ?? 0
  const deadChunk = data?.objects_by_status?.dead ?? 0
  const chunkedOk = data?.objects_by_status?.chunked ?? 0

  return (
    <div className="space-y-6">
      <PageHeader
        title="Queue"
        description={`Pipeline status, refreshed every ${REFRESH_MS / 1000}s. Stages run automatically on their own schedule (download → chunk → embed).`}
        actions={
          isFetching && !isLoading ? (
            <Badge tone="accent" className="shrink-0">updating</Badge>
          ) : null
        }
      />

      <ErrorState error={error} />

      {/* Above the counters on purpose: if nothing is running, the numbers
          below are a backlog rather than progress. */}
      <WorkerStrip query={workers} />

      {isLoading ? (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-[86px] rounded-xl" />)}
        </div>
      ) : data ? (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <Stat label="Documents" value={data.documents_total} />
            <Stat
              label="Pending download"
              value={data.pending_download}
              tone={data.pending_download > 0 ? 'warn' : 'ok'}
            />
            <Stat label="Pending chunk" value={pendingChunk} tone={pendingChunk > 0 ? 'warn' : 'ok'} />
            <Stat label="Chunked ok" value={chunkedOk} tone="ok" />
            {/* status='failed' now covers a failed download as well as a
                failed OCR; only the list below can tell them apart. */}
            <Stat label="Failed" value={failedChunk} tone={failedChunk > 0 ? 'warn' : 'default'} />
            <Stat
              label="Pending embed"
              value={data.chunks_pending_embed}
              tone={data.chunks_pending_embed > 0 ? 'warn' : 'ok'}
            />
          </div>

          {/* Separate from "failed": these are no longer being retried, so
              nothing will change unless someone looks. A counter that only
              ever goes up and never drains needs to say why. */}
          {deadChunk > 0 && (
            <p className="text-xs text-danger">
              {deadChunk} PDF{deadChunk !== 1 ? 's' : ''} gave up after repeated chunking
              failures and will not be retried.
            </p>
          )}

          <p className="text-xs text-faint">
            Embedding model: <code className="rounded-sm bg-inset px-1.5 py-0.5">{data.embed_model}</code>
          </p>
        </>
      ) : null}

      <section className="space-y-2">
        <h2 className="text-sm font-semibold">In the pipeline</h2>
        <ErrorState error={queue.error} />
        {queue.isLoading ? (
          <Card className="divide-y divide-border">
            {[0, 1, 2].map((i) => (
              <div key={i} className="p-3">
                <Skeleton className="h-4 w-2/3" />
              </div>
            ))}
          </Card>
        ) : !queue.data?.length ? (
          <EmptyState
            icon={Inbox}
            title="Nothing queued"
            description="No papers registered yet. Fetch some to get the pipeline moving."
          />
        ) : (
          <Card className="overflow-hidden p-0">
            {queue.data.map((item) => (
              <QueueRow key={`${item.doc_id}-${item.obj_id ?? 'none'}`} item={item} />
            ))}
          </Card>
        )}
      </section>
    </div>
  )
}
