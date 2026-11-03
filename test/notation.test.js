import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NUM_JOINTS, forwardKinematics, isValid, straight } from '../src/kinematics.js';
import { parseNotation, toNotation, jointOf, pieceOf } from '../src/notation.js';

// Examples from http://thomas-wolter.de/rubik_notation_en.html. The page
// writes the last Snowflake turn as "12R" without a position; the turns of
// the Snowflake repeat with period 4 (R3 L1, R3 L3, R1 L3, R1 L1), which
// makes it 12R1.
const WOLF = '2R2-3L2-4L2-5R2-6R2-7L2-8L2-10L2-10R2-12R2';
const CAT = '9R2-9L2-8L2-7R2-6R2-6L2-5L3-4L2-3R2-2R2-2L2';
const SNOWFLAKE = '1R3-2L1-2R3-3L3-3R1-4L3-4R1-5L1-5R3-6L1-6R3-7L3-7R1-8L3-8R1-9L1-9R3-10L1-10R3-12L3-11R1-11L3-12R1';

test('joints map to dark wedge and side', () => {
    assert.equal(jointOf(1, 'R'), 0);
    assert.equal(jointOf(2, 'L'), 1);
    assert.equal(jointOf(2, 'R'), 2);
    assert.equal(jointOf(12, 'L'), 21);
    assert.equal(jointOf(12, 'R'), 22);
    for (let j = 0; j < NUM_JOINTS; j++) {
        const { piece, side } = pieceOf(j);
        assert.equal(jointOf(piece, side), j);
    }
});

test('straight stick is the empty notation', () => {
    assert.deepEqual(parseNotation(''), straight());
    assert.deepEqual(parseNotation('  '), straight());
    assert.equal(toNotation(straight()), '');
});

test('single turns', () => {
    const a = parseNotation('11L1');
    assert.equal(a[jointOf(11, 'L')], 1);
    assert.equal(a.filter(x => x !== 0).length, 1);
    assert.equal(toNotation(a), '11L1');
    // Position 0 is the base position (used as "12R0" on the page).
    assert.deepEqual(parseNotation('12R0'), straight());
});

test('examples from the page are valid shapes', () => {
    for (const s of [WOLF, CAT, SNOWFLAKE]) {
        const a = parseNotation(s);
        assert.ok(isValid(a), s);
        assert.equal(a.filter(x => x !== 0).length, s.split('-').length);
    }
});

test('snowflake is closed and symmetric', () => {
    const poses = forwardKinematics(parseNotation(SNOWFLAKE));
    const pos = poses.map(p => p.position);
    // Ends meet: last wedge sits next to the first.
    const d = pos[23].map((v, i) => v - pos[0][i]);
    assert.equal(Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]), 1);
    // Occupies a 3x3x3 block without its centre and corners, the six face
    // centres holding two wedges each.
    const counts = new Map();
    for (const p of pos) counts.set(p.join(), (counts.get(p.join()) ?? 0) + 1);
    const lo = [0, 1, 2].map(i => Math.min(...pos.map(p => p[i])));
    for (const [k, n] of counts) {
        const off = k.split(',').map((v, i) => Number(v) - lo[i] - 1);
        const nonZero = off.filter(v => v !== 0).length;
        assert.ok(off.every(v => Math.abs(v) <= 1));
        assert.equal(n, nonZero === 1 ? 2 : 1, k);
        assert.notEqual(nonZero, 0);
        assert.notEqual(nonZero, 3);
    }
});

test('writing sorts by wedge, left before right', () => {
    assert.equal(toNotation(parseNotation(CAT)), '2L2-2R2-3R2-4L2-5L3-6L2-6R2-7R2-8L2-9L2-9R2');
    assert.equal(toNotation(parseNotation(WOLF)), WOLF);
    assert.equal(toNotation(parseNotation(SNOWFLAKE)),
        '1R3-2L1-2R3-3L3-3R1-4L3-4R1-5L1-5R3-6L1-6R3-7L3-7R1-8L3-8R1-9L1-9R3-10L1-10R3-11L3-11R1-12L3-12R1');
});

test('lenient separators and case', () => {
    assert.deepEqual(parseNotation('2r2, 3l2  4L2'), parseNotation('2R2-3L2-4L2'));
});

test('round trip for random angles', () => {
    let seed = 7;
    const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32;
    for (let n = 0; n < 200; n++) {
        const a = Array.from({ length: NUM_JOINTS }, () => Math.floor(rand() * 4));
        assert.deepEqual(parseNotation(toNotation(a)), a);
    }
});

test('other snake lengths', () => {
    assert.deepEqual(parseNotation('2L1-2R3', 4), [0, 1, 3]);
    assert.equal(toNotation([0, 1, 3]), '2L1-2R3');
    assert.throws(() => parseNotation('3R1', 4), /wedge must be 1..2/);
});

test('rejects malformed turns', () => {
    assert.throws(() => parseNotation('12R'), /invalid turn "12R"/);
    assert.throws(() => parseNotation('1L1'), /no left joint/);
    assert.throws(() => parseNotation('13R1'), /wedge must be/);
    assert.throws(() => parseNotation('0R1'), /wedge must be/);
    assert.throws(() => parseNotation('3R4'), /invalid turn/);
    assert.throws(() => parseNotation('3X1'), /invalid turn/);
    assert.throws(() => parseNotation('3R1-3R2'), /given twice/);
    assert.throws(() => toNotation([0.5]), /integer/);
});
