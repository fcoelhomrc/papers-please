import { PageHeader, Card, ErrorState, Skeleton, Badge } from '../components/ui.jsx'
import { useStatus } from '../hooks/queries'
import clsx from 'clsx'

const REFRESH_MS = 10_000

function Stat({ label, value, tone = 'default' }) {
  const toneClass = { default: 'text-ink', warn: 'text-warn', ok: 'text-success' }[tone]
  return (
    <Card className="p-4">
      <div className="text-2xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx('mt-1.5 text-3xl font-semibold tabular-nums', toneClass)}>{value}</div>
    </Card>
  )
}

export default function Queue() {
  const { data, error, isLoading, isFetching } = useStatus({ refetchInterval: REFRESH_MS })

  const pendingChunk = data?.objects_by_status?.pending ?? 0
  const failedChunk = data?.objects_by_status?.failed ?? 0
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
            <Stat label="Chunk failed" value={failedChunk} tone={failedChunk > 0 ? 'warn' : 'default'} />
            <Stat
              label="Pending embed"
              value={data.chunks_pending_embed}
              tone={data.chunks_pending_embed > 0 ? 'warn' : 'ok'}
            />
          </div>
          <p className="text-xs text-faint">
            Embedding model: <code className="rounded-sm bg-inset px-1.5 py-0.5">{data.embed_model}</code>
          </p>
        </>
      ) : null}
    </div>
  )
}
