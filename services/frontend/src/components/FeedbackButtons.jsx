import clsx from 'clsx'
import { ThumbsDown, ThumbsUp } from 'lucide-react'
import { useState } from 'react'
import { useFeedback } from '../hooks/queries'

/* Thumbs on a result or a citation.
 *
 * The eval set is grown by hand in eval/fixtures.py while every search
 * someone runs is a labelling opportunity going to waste. A thumbs-up is
 * exactly what `relevant_source_ids` encodes: for this question, this paper
 * is relevant. `eval/feedback.py` turns them into proposed dataset rows.
 *
 * Optimistic and unreversible-looking on purpose: the judgement is recorded
 * the moment it's clicked and the buttons then step out of the way. A
 * confirmation step on a thumb would cost more attention than the thumb is
 * worth, and nobody would ever click one.
 */
export default function FeedbackButtons({ query, docId, chunkId, kind = 'search', className }) {
  const [verdict, setVerdict] = useState(null)
  const feedback = useFeedback()

  function send(next) {
    if (verdict) return
    setVerdict(next)
    feedback.mutate(
      { kind, query, doc_id: docId, chunk_id: chunkId, verdict: next },
      // Revert on failure rather than leaving a judgement that looks
      // recorded and isn't - a silently dropped label is worse than none,
      // because it stops the person offering it again.
      { onError: () => setVerdict(null) },
    )
  }

  if (verdict) {
    return (
      <span className={clsx('text-2xs text-faint', className)}>
        {verdict === 'up' ? 'Marked relevant' : 'Marked not relevant'}
      </span>
    )
  }

  return (
    <span className={clsx('flex items-center gap-0.5', className)}>
      <button
        type="button"
        onClick={() => send('up')}
        aria-label="Mark this result relevant"
        title="Relevant — helps build the eval set"
        className="rounded p-1 text-faint transition-colors hover:bg-inset hover:text-accent"
      >
        <ThumbsUp size={12} />
      </button>
      <button
        type="button"
        onClick={() => send('down')}
        aria-label="Mark this result not relevant"
        title="Not relevant"
        className="rounded p-1 text-faint transition-colors hover:bg-inset hover:text-danger"
      >
        <ThumbsDown size={12} />
      </button>
    </span>
  )
}
