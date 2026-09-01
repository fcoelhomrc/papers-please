/* Run with: npm test  (node --test, built in - no test dependency) */
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { pdfUrl } from './urls.js'

test('plain preview url', () => {
  assert.equal(pdfUrl(7), '/api/documents/7/pdf')
})

test('a page becomes a viewer fragment', () => {
  assert.equal(pdfUrl(7, { page: 4 }), '/api/documents/7/pdf#page=4')
})

test('download forces the query parameter', () => {
  assert.equal(pdfUrl(7, { download: true }), '/api/documents/7/pdf?download=true')
})

test('the fragment follows the query string, never precedes it', () => {
  // '#page=4?download=true' would put the query inside the fragment and the
  // backend would never see it.
  assert.equal(
    pdfUrl(7, { download: true, page: 4 }),
    '/api/documents/7/pdf?download=true#page=4',
  )
})

test('a missing page adds no fragment', () => {
  assert.equal(pdfUrl(7, { page: null }), '/api/documents/7/pdf')
  assert.equal(pdfUrl(7, { page: undefined }), '/api/documents/7/pdf')
})

test('page 0 is not a page', () => {
  // PDF viewers are 1-indexed; a 0 here means "unknown", not "the first page".
  assert.equal(pdfUrl(7, { page: 0 }), '/api/documents/7/pdf')
})
