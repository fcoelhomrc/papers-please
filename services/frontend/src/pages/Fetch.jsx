import { useState } from 'react'
import {
  Button,
  Callout,
  Card,
  ErrorState,
  Field,
  Input,
  PageHeader,
} from '../components/ui.jsx'
import { useFetchPapers } from '../hooks/queries'

export default function Fetch() {
  const [form, setForm] = useState({ query: '', venue: '', year: '', maxPapers: 500 })
  const mutation = useFetchPapers()

  const set = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }))

  return (
    <div className="max-w-xl space-y-6">
      <PageHeader
        title="Fetch papers"
        description="Query Semantic Scholar and register new papers for download and indexing. Papers already in the library are skipped automatically."
      />

      <Card className="p-5">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            mutation.mutate({
              query: form.query,
              venue: form.venue,
              year: form.year,
              maxPapers: Number(form.maxPapers),
            })
          }}
          className="space-y-4"
        >
          <Field label="Query">
            <Input value={form.query} onChange={set('query')} placeholder="e.g. attention transformers" />
          </Field>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Venue">
              <Input value={form.venue} onChange={set('venue')} placeholder="e.g. NeurIPS" />
            </Field>
            <Field label="Year">
              <Input value={form.year} onChange={set('year')} placeholder="e.g. 2023 or 2020-2024" />
            </Field>
          </div>

          <Field label="Max papers" hint="Upper bound on results requested from Semantic Scholar.">
            <Input type="number" value={form.maxPapers} onChange={set('maxPapers')} min={1} max={5000} />
          </Field>

          <Button
            type="submit"
            variant="primary"
            size="lg"
            loading={mutation.isPending}
            className="w-full justify-center"
          >
            {mutation.isPending ? 'Fetching…' : 'Fetch'}
          </Button>
        </form>
      </Card>

      <ErrorState error={mutation.error} />

      {mutation.isSuccess && (
        <Callout tone="success">
          Added <strong>{mutation.data.fetched}</strong> new papers — duplicates already in the
          library were skipped. The pipeline will download and index them shortly.
        </Callout>
      )}
    </div>
  )
}
