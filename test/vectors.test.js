import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildVectors } from '../scripts/export-vectors.js';

// The Python port (sim/) is checked against test/vectors.json. If this test
// fails after a change to the model, regenerate the file with
// `node scripts/export-vectors.js` and run the Python tests.
test('test/vectors.json matches the current JS model', () => {
    const committed = JSON.parse(readFileSync(new URL('./vectors.json', import.meta.url), 'utf8'));
    assert.deepEqual(buildVectors(), committed);
});
