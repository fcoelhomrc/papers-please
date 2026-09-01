/* Splitting a stored chunk back into its section breadcrumb and its body.
 *
 * Chunks are stored as "Methods > Ablations\n\n<body>" - the section path is
 * prepended before embedding so the vector knows where in the paper the text
 * sits (see the backend's process/chunker.py). That prefix is genuinely part
 * of the indexed text, but rendering it inline inside the quotation makes it
 * read as the passage's first sentence, which it is not.
 *
 * Splitting it back out is presentation undoing an indexing decision, on
 * purpose. Kept in a plain .js module rather than beside the component so it
 * can be exercised by `node --test` without a JSX transform.
 */

// A section path is short. Beyond this it is far more likely to be a real
// opening paragraph that happens to be followed by a blank line.
export const MAX_HEADING_CHARS = 120

export function splitHeading(text) {
  if (!text) return { heading: null, body: '' }

  const [first, ...rest] = text.split('\n\n')
  // Conservative on all three counts: there has to be something after it,
  // it has to be a single line, and it has to be short. A passage that
  // merely starts with a brief sentence must not be relabelled as a heading
  // and then rendered in small caps as though it were structural.
  const looksLikeHeading =
    rest.length > 0 && !first.includes('\n') && first.length <= MAX_HEADING_CHARS

  return looksLikeHeading
    ? { heading: first.trim(), body: rest.join('\n\n') }
    : { heading: null, body: text }
}
