import { useEffect, useState } from 'react'
import { getStatus } from '../api'

const REFRESH_MS = 10_000

function StatCard({ label, value, tone = 'default' }) {
  const toneClass = {
    default: 'text-ink',
    warn: 'text-amber-600 dark:text-amber-400',
    ok: 'text-green-700 dark:text-green-400',
  }[tone]

  return (
    <div className="bg-surface border border-border rounded-xl p-4">
      <div className="text-[11.5px] text-muted uppercase tracking-wide">{label}</div>
      <div className={`text-2xl font-semibold mt-1 tabular-nums ${toneClass}`}>{value}</div>
    </div>
  )
}

export default function Queue() {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    function load() {
      getStatus()
        .then((data) => { if (!cancelled) { setStatus(data); setError(null) } })
        .catch((err) => { if (!cancelled) setError(err.message) })
    }
    load()
    const id = setInterval(load, REFRESH_MS)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const pendingChunk = status?.objects_by_status?.pending ?? 0
  const failedChunk = status?.objects_by_status?.failed ?? 0
  const chunkedOk = status?.objects_by_status?.chunked ?? 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Queue</h1>
        <p className="text-[13px] text-muted mt-0.5">
          Pipeline status, refreshed every {REFRESH_MS / 1000}s. Stages run automatically on
          their own schedule (download → chunk → embed).
        </p>
      </div>

      {error && <p className="text-[13px] text-red-600 bg-red-50 border border-red-200 rounded-lg px-4 py-2.5">{error}</p>}

      {!status && !error ? (
        <p className="text-[13px] text-faint">Loading…</p>
      ) : status ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <StatCard label="Documents" value={status.documents_total} />
            <StatCard label="Pending download" value={status.pending_download} tone={status.pending_download > 0 ? 'warn' : 'ok'} />
            <StatCard label="Pending chunk" value={pendingChunk} tone={pendingChunk > 0 ? 'warn' : 'ok'} />
            <StatCard label="Chunked ok" value={chunkedOk} tone="ok" />
            <StatCard label="Chunk failed" value={failedChunk} tone={failedChunk > 0 ? 'warn' : 'default'} />
            <StatCard label="Pending embed" value={status.chunks_pending_embed} tone={status.chunks_pending_embed > 0 ? 'warn' : 'ok'} />
          </div>
          <p className="text-[12px] text-faint">
            Embedding model: <code>{status.embed_model}</code>
          </p>
        </>
      ) : null}
    </div>
  )
}
