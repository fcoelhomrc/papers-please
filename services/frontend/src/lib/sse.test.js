/* Run with: npm test  (node --test, built in - no test dependency) */
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createSSEParser } from './sse.js'

test('parses a complete event', () => {
  const feed = createSSEParser()
  assert.deepEqual(feed('event: step\ndata: {"kind":"tool_call"}\n\n'), [
    { event: 'step', data: { kind: 'tool_call' } },
  ])
})

test('parses several events from one chunk', () => {
  const feed = createSSEParser()
  const got = feed('event: step\ndata: {"n":1}\n\nevent: step\ndata: {"n":2}\n\n')
  assert.deepEqual(got.map((e) => e.data.n), [1, 2])
})

test('an event split across chunks is not lost', () => {
  // The case that makes this a module: chunk boundaries land wherever the
  // network puts them, routinely inside the JSON.
  const feed = createSSEParser()
  assert.deepEqual(feed('event: step\ndata: {"ki'), [])
  assert.deepEqual(feed('nd":"tool_call"}\n\n'), [
    { event: 'step', data: { kind: 'tool_call' } },
  ])
})

test('a chunk boundary between the event and data lines', () => {
  const feed = createSSEParser()
  assert.deepEqual(feed('event: done\n'), [])
  assert.deepEqual(feed('data: {"reply":"hi"}\n\n'), [
    { event: 'done', data: { reply: 'hi' } },
  ])
})

test('holds a partial trailing event until it completes', () => {
  const feed = createSSEParser()
  const got = feed('event: step\ndata: {"n":1}\n\nevent: step\ndata: {"n":2}')
  assert.equal(got.length, 1)
  assert.deepEqual(feed('\n\n')[0].data, { n: 2 })
})

test('defaults to the message event when none is named', () => {
  const feed = createSSEParser()
  assert.equal(feed('data: {"a":1}\n\n')[0].event, 'message')
})

test('one malformed event does not kill the stream', () => {
  const feed = createSSEParser()
  const got = feed('event: step\ndata: not json\n\nevent: done\ndata: {"ok":true}\n\n')
  assert.deepEqual(got, [{ event: 'done', data: { ok: true } }])
})

test('ignores comment lines', () => {
  const feed = createSSEParser()
  assert.deepEqual(feed(': keep-alive\nevent: step\ndata: {"n":1}\n\n')[0].data, { n: 1 })
})

test('multi-line data is joined', () => {
  const feed = createSSEParser()
  assert.deepEqual(feed('event: step\ndata: {"a":\ndata: 1}\n\n')[0].data, { a: 1 })
})
