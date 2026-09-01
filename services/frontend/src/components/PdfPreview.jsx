import * as Dialog from '@radix-ui/react-dialog'
import { Download, FileText, X } from 'lucide-react'
import { pdfUrl } from '../api'
import { Button, buttonClass } from './ui.jsx'

export default function PdfPreview({ docId, title, page = null, label = 'Preview', size = 'sm' }) {
  // Inline for the iframe (renders in place); download=true forces the
  // browser to save the file instead - the backend defaults to inline
  // Content-Disposition, which is what makes this button distinct.
  //
  // `page` lands as a #page=N fragment. Every result already knew which page
  // its passage came from and the preview still opened at page 1, leaving
  // the reader to go find it.
  const previewUrl = pdfUrl(docId, { page })
  const downloadUrl = pdfUrl(docId, { download: true })

  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <Button size={size}>
          <FileText size={13} /> {label}
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6">
          <div className="flex h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-panel animate-slide-up">
            <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border px-4">
              <Dialog.Title className="truncate text-sm font-medium">
                {title}
                {page != null && <span className="ml-2 text-xs text-muted">page {page}</span>}
              </Dialog.Title>
              <div className="flex shrink-0 items-center gap-2">
                <a href={downloadUrl} className={buttonClass({ size: 'sm' })}>
                  <Download size={13} /> Download
                </a>
                <Dialog.Close asChild>
                  <Button variant="ghost" size="icon" aria-label="Close preview">
                    <X size={15} />
                  </Button>
                </Dialog.Close>
              </div>
            </div>
            {/* Keyed on the URL so reopening at a different page remounts the
                iframe. Browsers ignore a fragment-only src change on a live
                iframe, so without this the second preview would silently
                stay on the first one's page. */}
            <iframe
              key={previewUrl}
              src={previewUrl}
              title={title}
              className="w-full flex-1 bg-inset"
            />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
