import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findCollisions, isValid, twist, validMoves, straight } from '../src/kinematics.js';
import {
    wedgeShape, shapesIntersect, findGeometricCollisions, findSweepCollision, isTwistFree,
} from '../src/sweep.js';

// Deterministic PRNG so failures are reproducible.
function rng(seed) {
    return () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 2 ** 32;
    };
}

const ID = [1, 0, 0, 0, 1, 0, 0, 0, 1];
// Rotation by 180 degrees about the X axis: maps the half cube y + z <= 0 onto y + z >= 0.
const FLIP_X = [1, 0, 0, 0, -1, 0, 0, 0, -1];

test('a wedge and its complement in the same cell do not intersect', () => {
    const a = wedgeShape({ position: [0, 0, 0], rotation: ID });
    const b = wedgeShape({ position: [0, 0, 0], rotation: FLIP_X });
    assert.equal(shapesIntersect(a, b), false);
    assert.equal(shapesIntersect(a, a), true);
});

test('wedges in face-adjacent cells only touch', () => {
    const a = wedgeShape({ position: [0, 0, 0], rotation: ID });
    for (const p of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]]) {
        assert.equal(shapesIntersect(a, wedgeShape({ position: p, rotation: ID })), false, p.join(','));
    }
    // Shifted by a fraction of the cell they overlap.
    assert.equal(shapesIntersect(a, wedgeShape({ position: [0.3, 0, 0], rotation: ID })), true);
});

test('geometric check agrees with the lattice check on integer configurations', () => {
    const rand = rng(7);
    let colliding = 0;
    for (let n = 0; n < 500; n++) {
        const angles = Array.from({ length: 23 }, () => Math.floor(rand() * 4));
        // The lattice check also lists directly connected pairs, which never overlap.
        const lattice = findCollisions(angles).filter(([i, j]) => j - i >= 2);
        assert.deepEqual(findGeometricCollisions(angles), lattice, angles.join(''));
        if (lattice.length) colliding++;
    }
    assert.ok(colliding > 100, 'sample should contain many colliding configurations');
});

test('every single twist of the straight stick is free', () => {
    const s = straight();
    for (let joint = 0; joint < s.length; joint++) {
        for (const direction of [1, -1, 2]) {
            assert.equal(findSweepCollision(s, joint, direction), null, `${joint} ${direction}`);
        }
    }
});

test('a twist can be blocked on the way although its end state is valid', () => {
    // Joint 1 at a half turn: wedges 0 and 3 meet along an edge, so turning
    // joint 0 drives wedge 3 into wedge 0 in either direction.
    const angles = straight();
    angles[1] = 2;
    for (const direction of [1, -1]) {
        assert.ok(isValid(twist(angles, 0, direction)));
        const hit = findSweepCollision(angles, 0, direction);
        assert.deepEqual(hit && hit.pair, [0, 3]);
        assert.equal(isTwistFree(angles, 0, direction), false);
    }
});

test('a twist and its reverse are blocked alike', () => {
    const rand = rng(3);
    let angles = straight();
    let blocked = 0;
    for (let s = 0; s < 150; s++) {
        const moves = validMoves(angles);
        const m = moves[Math.floor(rand() * moves.length)];
        const there = findSweepCollision(angles, m.joint, m.direction) === null;
        const after = twist(angles, m.joint, m.direction);
        const back = findSweepCollision(after, m.joint, -m.direction) === null;
        assert.equal(there, back, `${angles.join('')} joint ${m.joint} dir ${m.direction}`);
        if (there) angles = after; else blocked++;
    }
    assert.ok(blocked > 0, 'walk should run into blocked twists');
});

test('only wedges on opposite sides of the joint are reported', () => {
    const rand = rng(11);
    let angles = straight();
    for (let s = 0; s < 200; s++) {
        const moves = validMoves(angles);
        const m = moves[Math.floor(rand() * moves.length)];
        const hit = findSweepCollision(angles, m.joint, m.direction);
        if (hit) {
            assert.ok(hit.pair[0] <= m.joint && hit.pair[1] > m.joint);
            assert.ok(hit.fraction > 0 && hit.fraction <= 1);
        } else {
            angles = twist(angles, m.joint, m.direction);
        }
    }
});
