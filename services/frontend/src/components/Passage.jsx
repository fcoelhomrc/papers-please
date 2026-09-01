import clsx from 'clsx'
import { splitHeading } from '../lib/passage.js'

/* A retrieved chunk, rendered as a section breadcrumb plus the passage.
 *
 * The split itself lives in lib/passage.js so it can be tested without a JSX
 * transform; see that file for why the stored text has a heading prefix at
 * all and why the split is deliberately conservative.
 */
export default function Passage({ text, className }) {
  const { heading, body } = splitHeading(text)

  return (
    <div className={clsx('rounded-r-lg border-l-2 border-accent/40 bg-inset px-3 py-2', className)}>
      {heading && (
        <p className="mb-1 text-2xs font-medium uppercase tracking-wide text-faint">{heading}</p>
      )}
      <blockquote className="whitespace-pre-wrap text-sm leading-relaxed">{body}</blockquote>
    </div>
  )
}
