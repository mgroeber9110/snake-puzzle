// Exports test vectors from the JS reference model (kinematics.js, sweep.js,
// notation.js) so the Python port in sim/ can be checked against it.
//
//   node scripts/export-vectors.js            writes test/vectors.json
//   node scripts/export-vectors.js --stdout   prints the vectors instead
//
// The vectors are deterministic (fixed seeds); test/vectors.test.js checks that
// the committed file matches what the current model produces.

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
    forwardKinematics, findCollisions, validMoves, twist, straight, isValid,
} from '../src/kinematics.js';
import { isTwistFree } from '../src/sweep.js';
import { parseNotation, toNotation } from '../src/notation.js';

function rng(seed) {
    return () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 2 ** 32;
    };
}

// Random walk of valid moves from the straight stick.
function randomWalk(rand, numWedges, steps) {
    let angles = straight(numWedges);
    for (let k = 0; k < steps; k++) {
        const moves = validMoves(angles);
        const m = moves[Math.floor(rand() * moves.length)];
        angles = twist(angles, m.joint, m.direction);
    }
    return angles;
}

function record(angles, { sweep }) {
    const poses = forwardKinematics(angles);
    const v = {
        angles,
        positions: poses.map(p => p.position),
        rotations: poses.map(p => p.rotation),
        collisions: findCollisions(angles),
        valid: isValid(angles),
    };
    if (v.valid) {
        v.validMoves = validMoves(angles).map(m => [m.joint, m.direction]);
        if (sweep) {
            v.freeMoves = v.validMoves.filter(([j, d]) => isTwistFree(angles, j, d));
        }
    }
    if ((angles.length + 1) % 2 === 0) v.notation = toNotation(angles);
    return v;
}

export const SHAPES = {
    wolf: '2R2-3L2-4L2-5R2-6R2-7L2-8L2-10L2-10R2-12R2',
    cat: '9R2-9L2-8L2-7R2-6R2-6L2-5L3-4L2-3R2-2R2-2L2',
    ball: '1R1-2L3-2R3-3L1-3R1-4L1-4R1-5L3-5R3-12R3-12L3-11R3-11L3-10R1-10L1-9R1-9L1-8R3-8L3-7R1-6R3-6L3-7L1',
    snowflake: '1R3-2L1-2R3-3L3-3R1-4L3-4R1-5L1-5R3-6L1-6R3-7L3-7R1-8L3-8R1-9L1-9R3-10L1-10R3-12L3-11R1-11L3-12R1',
    cobra: '2L3-2R3-3L1-3R1-4L1-5L3-4R1-5R3-6L3-7L1-6R3-7R3-8L1-9L1-9R1-11L2-12R1',
};

export function buildVectors() {
    const configs = [];
    const rand = rng(12345);
    // Arbitrary (mostly invalid) configurations: forward kinematics and collisions.
    for (const n of [8, 12, 24]) {
        for (let k = 0; k < 40; k++) {
            const angles = Array.from({ length: n - 1 }, () => Math.floor(rand() * 4));
            configs.push(record(angles, { sweep: false }));
        }
    }
    // Valid configurations from random walks, a few with the sweep check (slow).
    for (const [n, steps, count, sweepCount] of [[8, 6, 20, 10], [12, 10, 20, 8], [24, 30, 30, 6]]) {
        for (let k = 0; k < count; k++) {
            configs.push(record(randomWalk(rand, n, steps), { sweep: k < sweepCount }));
        }
    }
    for (const n of [8, 24]) configs.push(record(straight(n), { sweep: true }));

    const shapes = {};
    for (const [name, text] of Object.entries(SHAPES)) {
        const angles = parseNotation(text);
        shapes[name] = { notation: text, angles, canonical: toNotation(angles), valid: isValid(angles) };
    }
    return { generator: 'scripts/export-vectors.js', configs, shapes };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const json = JSON.stringify(buildVectors());
    if (process.argv.includes('--stdout')) process.stdout.write(json);
    else writeFileSync(new URL('../test/vectors.json', import.meta.url), json + '\n');
}
