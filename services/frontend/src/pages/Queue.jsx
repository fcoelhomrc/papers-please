export default function Queue() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-semibold">Queue</h1>
        <p className="text-[13px] text-muted mt-0.5">Pipeline status dashboard — coming soon.</p>
      </div>
      <p className="text-[13px] text-faint">
        Will show documents pending download/chunk/embed and per-stage counts once the backend
        exposes a status endpoint.
      </p>
    </div>
  )
}
