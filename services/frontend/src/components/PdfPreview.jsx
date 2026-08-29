import * as Dialog from '@radix-ui/react-dialog'
import { Download, FileText, X } from 'lucide-react'
import { pdfUrl } from '../api'
import { Button, buttonClass } from './ui.jsx'

export default function PdfPreview({ docId, title }) {
  // Inline for the iframe (renders in place); download=true forces the
  // browser to save the file instead - the backend defaults to inline
  // Content-Disposition, which is what makes this button distinct.
  const previewUrl = pdfUrl(docId)
  const downloadUrl = pdfUrl(docId, { download: true })

  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <Button size="sm">
          <FileText size={13} /> Preview
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 animate-fade-in" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6">
          <div className="flex h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-panel animate-slide-up">
            <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border px-4">
              <Dialog.Title className="truncate text-sm font-medium">{title}</Dialog.Title>
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
            <iframe src={previewUrl} title={title} className="w-full flex-1 bg-inset" />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
