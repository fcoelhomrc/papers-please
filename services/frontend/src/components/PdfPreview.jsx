import * as Dialog from '@radix-ui/react-dialog'
import { Download, FileText, X } from 'lucide-react'
import { pdfUrl } from '../api'

export default function PdfPreview({ docId, title }) {
  const url = pdfUrl(docId)
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <button className="inline-flex items-center gap-1.5 text-xs font-medium border border-border rounded-md px-2.5 py-1 hover:border-faint transition-colors">
          <FileText size={13} /> Preview
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 bg-black/40 z-40" />
        <Dialog.Content className="fixed inset-0 z-50 flex items-center justify-center p-6">
          <div className="bg-surface border border-border rounded-xl shadow-2xl w-full max-w-3xl h-[85vh] flex flex-col overflow-hidden">
            <div className="h-12 flex items-center justify-between px-4 border-b border-border shrink-0">
              <Dialog.Title className="text-[13.5px] font-medium truncate pr-4">{title}</Dialog.Title>
              <div className="flex items-center gap-3 shrink-0">
                <a
                  href={url}
                  download
                  className="inline-flex items-center gap-1.5 text-xs font-medium border border-border rounded-md px-2.5 py-1 hover:border-faint transition-colors"
                >
                  <Download size={13} /> Download
                </a>
                <Dialog.Close asChild>
                  <button className="text-muted hover:text-ink" aria-label="Close preview">
                    <X size={16} />
                  </button>
                </Dialog.Close>
              </div>
            </div>
            <iframe src={url} title={title} className="flex-1 w-full" />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
