/* Server-sent-event parsing for the agent stream.
 *
 * A plain module, not part of api.js, for two reasons: it is the only piece
 * of the streaming client with edge cases worth testing (a network chunk can
 * split an event anywhere, including mid-line), and it needs no fetch to
 * exercise.
 *
 * Deliberately not EventSource, which only does GET and cannot send a JSON
 * body - the chat turn needs POST.
 */

/* Feeds raw text chunks in, gets complete {event, data} objects out.
 *
 * The buffer is the whole point. A chunk boundary lands wherever the network
 * puts it, so "event: step\ndata: {...}" routinely arrives as two chunks
 * with the split inside the JSON. Parsing each chunk independently would
 * throw on the fragment and lose the event.
 */
export function createSSEParser() {
  let buffer = ''

  return function feed(chunk) {
    buffer += chunk
    const events = []

    // A blank line terminates an event. Anything after the last one is a
    // partial event and stays in the buffer for the next chunk.
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() ?? ''

    for (const block of blocks) {
      let name = 'message'
      const dataLines = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) name = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
        // ':' comment lines and unknown fields are ignored, per the spec.
      }
      if (!dataLines.length) continue
      try {
        events.push({ event: name, data: JSON.parse(dataLines.join('\n')) })
      } catch {
        // A server that emitted non-JSON data is a bug, but dropping one
        // malformed event beats tearing down a stream that is still
        // delivering good ones.
      }
    }
    return events
  }
}
