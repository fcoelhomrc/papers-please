/* Run with: npm test  (node --test, built in - no test dependency) */
import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  citationIndex,
  hasFailure,
  isAbstention,
  linkCitations,
  parseCitationHref,
} from './citations.js'

const evidence = [
  { doc_id: 3, chunk_id: 118, title: 'First' },
  { doc_id: 7, chunk_id: 204, title: 'Second' },
]

test('numbers cards from one, not by doc_id', () => {
  // "[147]" in running text reads as a footnote to a bibliography the
  // reader doesn't have.
  assert.deepEqual([...citationIndex(evidence)], [[3, 1], [7, 2]])
})

test('a doc cited twice keeps one number', () => {
  const index = citationIndex([{ doc_id: 3 }, { doc_id: 3 }, { doc_id: 7 }])
  assert.deepEqual([...index], [[3, 1], [7, 2]])
})

test('rewrites a citation into a numbered link', () => {
  assert.equal(
    linkCitations('Yes, per [doc 3, p4].', evidence),
    'Yes, per [1](citation:3:4).',
  )
})

test('handles the shapes a model actually writes', () => {
  const cases = [
    ['[doc 3, p4]', '[1](citation:3:4)'],
    ['[doc 3, p. 4]', '[1](citation:3:4)'],
    ['[doc 3, page 4]', '[1](citation:3:4)'],
    ['[Doc 3, Page 4]', '[1](citation:3:4)'],
    ['[doc 3]', '[1](citation:3)'],
    ['[doc. 3, pp. 4-5]', '[1](citation:3:4)'],
    ['[ doc 3 , p 4 ]', '[1](citation:3:4)'],
  ]
  for (const [input, want] of cases) {
    assert.equal(linkCitations(input, evidence), want, input)
  }
})

test('rewrites every citation in a paragraph', () => {
  assert.equal(
    linkCitations('One [doc 3, p4] and two [doc 7, p2].', evidence),
    'One [1](citation:3:4) and two [2](citation:7:2).',
  )
})

test('leaves a citation to a doc that was never retrieved untouched', () => {
  // Hiding it would conceal the most useful signal in the panel: the model
  // citing something it did not actually find.
  assert.equal(
    linkCitations('Claimed in [doc 99, p1].', evidence),
    'Claimed in [doc 99, p1].',
  )
})

test('leaves ordinary markdown links alone', () => {
  const text = 'See [the paper](https://example.com) for detail.'
  assert.equal(linkCitations(text, evidence), text)
})

test('empty input is not an error', () => {
  assert.equal(linkCitations('', evidence), '')
  assert.equal(linkCitations(undefined, evidence), '')
  assert.equal(linkCitations('[doc 3, p4]', []), '[doc 3, p4]')
})

test('parses a citation href back', () => {
  assert.deepEqual(parseCitationHref('citation:3:4'), { docId: 3, page: 4 })
  assert.deepEqual(parseCitationHref('citation:3'), { docId: 3, page: null })
  assert.equal(parseCitationHref('https://example.com'), null)
  assert.equal(parseCitationHref(''), null)
})

test('abstention is a successful search that found nothing', () => {
  const trace = [{ tool: 'search_chunks', ok: true, summary: '0 chunks' }]
  assert.equal(isAbstention(trace, []), true)
})

test('a failed search is not an abstention', () => {
  // One means "the library doesn't cover this", the other means "try again".
  const trace = [{ tool: 'search_chunks', ok: false, summary: 'pool exhausted' }]
  assert.equal(isAbstention(trace, []), false)
  assert.equal(hasFailure(trace), true)
})

test('a search that found something is not an abstention', () => {
  const trace = [{ tool: 'search_chunks', ok: true, summary: '2 chunks' }]
  assert.equal(isAbstention(trace, evidence), false)
})

test('a turn that never searched is not an abstention', () => {
  // A fetch request has no citations either, and labelling it "nothing
  // relevant found" would be nonsense.
  const trace = [{ tool: 'fetch_papers', ok: true, summary: 'fetched 50 papers' }]
  assert.equal(isAbstention(trace, []), false)
})

test('a turn with no tools at all is not an abstention', () => {
  assert.equal(isAbstention([], []), false)
})
