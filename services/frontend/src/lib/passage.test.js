/* Run with: npm test  (node --test, built in - no test dependency) */
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { splitHeading } from './passage.js'

test('splits a stored heading path off the body', () => {
  assert.deepEqual(splitHeading('Methods > Ablations\n\nWe ablate the encoder.'), {
    heading: 'Methods > Ablations',
    body: 'We ablate the encoder.',
  })
})

test('leaves a passage with no heading alone', () => {
  const text = 'Just a plain passage with no section prefix.'
  assert.deepEqual(splitHeading(text), { heading: null, body: text })
})

test('keeps every paragraph of the body', () => {
  assert.equal(splitHeading('Results\n\npara one\n\npara two').body, 'para one\n\npara two')
})

test('a long opening paragraph is not mistaken for a heading', () => {
  const long = 'A'.repeat(200)
  assert.equal(splitHeading(`${long}\n\nbody`).heading, null)
})

test('a multi-line opening block is not a heading', () => {
  assert.equal(splitHeading('line one\nline two\n\nbody').heading, null)
})

test('a short passage with no blank line is body, not a heading', () => {
  assert.deepEqual(splitHeading('Short.'), { heading: null, body: 'Short.' })
})

test('trims whitespace around the heading', () => {
  assert.equal(splitHeading('  Results  \n\nbody').heading, 'Results')
})

test('empty and missing text are not errors', () => {
  assert.deepEqual(splitHeading(''), { heading: null, body: '' })
  assert.deepEqual(splitHeading(undefined), { heading: null, body: '' })
})
