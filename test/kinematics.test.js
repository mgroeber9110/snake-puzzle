import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
    NUM_WEDGES, NUM_JOINTS, forwardKinematics, entryDirection, exitDirection,
    findCollisions, isValid, twist, validMoves, straight, normalizeAngles,
} from '../src/kinematics.js';

// Deterministic PRNG so failures are reproducible.
function rng(seed) {
    return () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 2 ** 32;
    };
}

function randomAngles(rand, joints) {
    return Array.from({ length: joints }, () => Math.floor(rand() * 4));
}

// Valid configurations reached by a random walk of valid moves from the stick.
function randomValidWalk(rand, numWedges, steps) {
    let angles = straight(numWedges);
    const seen = [];
    for (let s = 0; s < steps; s++) {
        const moves = validMoves(angles);
        const m = moves[Math.floor(rand() * moves.length)];
        angles = twist(angles, m.joint, m.direction);
        seen.push(angles);
    }
    return seen;
}

function det(m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// Independent reference: sample interior points of each wedge (directly from
// the half-cube definition y + z < 0 in local coordinates) and test whether
// any lies strictly inside another wedge.
function bruteForceCollisions(angles, n = 4) {
    const poses = forwardKinematics(angles);
    const offsets = [];
    for (let k = 0; k < n; k++) offsets.push((k + 0.5) / n - 0.5);
    const inside = (pose, p) => {
        const d = [p[0] - pose.position[0], p[1] - pose.position[1], p[2] - pose.position[2]];
        const r = pose.rotation;
        // local = R^T d
        const l = [
            r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
            r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
            r[2] * d[0] + r[5] * d[1] + r[8] * d[2],
        ];
        const eps = 1e-9;
        return Math.abs(l[0]) < 0.5 - eps && Math.abs(l[1]) < 0.5 - eps &&
               Math.abs(l[2]) < 0.5 - eps && l[1] + l[2] < -eps;
    };
    const pairs = [];
    for (let i = 0; i < poses.length; i++) {
        for (let j = i + 1; j < poses.length; j++) {
            let hit = false;
            for (const x of offsets) for (const y of offsets) for (const z of offsets) {
                if (hit || y + z >= 0) continue;
                const r = poses[i].rotation;
                const p = [
                    poses[i].position[0] + r[0] * x + r[1] * y + r[2] * z,
                    poses[i].position[1] + r[3] * x + r[4] * y + r[5] * z,
                    poses[i].position[2] + r[6] * x + r[7] * y + r[8] * z,
                ];
                if (inside(poses[j], p)) hit = true;
            }
            if (hit) pairs.push([i, j]);
        }
    }
    return pairs;
}

test('standard snake has 24 wedges and 23 joints', () => {
    assert.equal(NUM_WEDGES, 24);
    assert.equal(NUM_JOINTS, 23);
    assert.equal(straight().length, 23);
});

test('straight stick is a diagonal staircase and valid', () => {
    const poses = forwardKinematics(straight());
    poses.forEach((pose, i) => {
        assert.deepEqual(pose.position, [0, Math.floor(i / 2), 0 - Math.ceil(i / 2)]);
    });
    assert.ok(isValid(straight()));
});

test('poses are proper rotations and neighbours share a square face', () => {
    const rand = rng(1);
    for (let t = 0; t < 200; t++) {
        const poses = forwardKinematics(randomAngles(rand, NUM_JOINTS));
        for (const p of poses) {
            assert.equal(det(p.rotation), 1);
            for (const v of [...p.rotation, ...p.position]) assert.ok(Number.isInteger(v));
        }
        for (let i = 0; i + 1 < poses.length; i++) {
            const step = poses[i + 1].position.map((v, k) => v - poses[i].position[k]);
            const exit = exitDirection(poses[i]);
            assert.deepEqual(step, exit);
            assert.deepEqual(entryDirection(poses[i + 1]).map(v => 0 - v), exit);
        }
    }
});

test('twisting a joint keeps the shared face fixed and rotates by 90 degrees', () => {
    const base = straight();
    const j = 7;
    for (const q of [0.25, 0.5, 1, 2, 3]) {
        const a = forwardKinematics(base);
        const b = forwardKinematics(twist(base, j, 0).map((v, k) => (k === j ? v + q : v)));
        for (let i = 0; i <= j + 1; i++) {
            for (let k = 0; k < 3; k++) {
                assert.ok(Math.abs(a[i].position[k] - b[i].position[k]) < 1e-12);
            }
        }
        // Exit direction of the wedge before the joint is the axis; next entry stays opposite.
        const axis = exitDirection(b[j]);
        const entry = entryDirection(b[j + 1]);
        for (let k = 0; k < 3; k++) assert.ok(Math.abs(axis[k] + entry[k]) < 1e-12);
    }
});

test('fractional angles approach the integer pose continuously', () => {
    const rand = rng(2);
    const angles = randomAngles(rand, NUM_JOINTS);
    const exact = forwardKinematics(angles);
    const near = forwardKinematics(angles.map(a => a + 1e-9));
    exact.forEach((p, i) => {
        for (let k = 0; k < 9; k++) assert.ok(Math.abs(p.rotation[k] - near[i].rotation[k]) < 1e-6);
        for (let k = 0; k < 3; k++) assert.ok(Math.abs(p.position[k] - near[i].position[k]) < 1e-6);
    });
});

test('collision check matches brute-force sampling on random configurations', () => {
    const rand = rng(3);
    let colliding = 0;
    for (let t = 0; t < 300; t++) {
        const joints = 3 + Math.floor(rand() * 21);
        const angles = randomAngles(rand, joints);
        const fast = findCollisions(angles);
        assert.deepEqual(fast, bruteForceCollisions(angles), `angles ${angles}`);
        if (fast.length) colliding++;
    }
    assert.ok(colliding > 50, 'sample should contain many collisions');
});

test('collision check matches brute force on valid random walks, including shared cells', () => {
    const rand = rng(4);
    let sharedCells = 0;
    for (const angles of randomValidWalk(rand, NUM_WEDGES, 400)) {
        assert.deepEqual(bruteForceCollisions(angles), []);
        const cells = new Set(forwardKinematics(angles).map(p => p.position.join()));
        if (cells.size < NUM_WEDGES) sharedCells++;
    }
    assert.ok(sharedCells > 0, 'walk should reach shapes where two wedges form one cube');
});

test('four wedges close a ring; a fifth lands on the first wedge', () => {
    assert.ok(isValid([2, 2, 2]));
    for (const last of [0, 1, 2, 3]) {
        assert.deepEqual(findCollisions([2, 2, 2, last]), [[0, 4]]);
    }
});

test('collision check rejects fractional angles', () => {
    assert.throws(() => findCollisions([0.5, 0]));
});

test('validMoves and normalizeAngles', () => {
    const moves = validMoves(straight());
    assert.equal(moves.length, 2 * NUM_JOINTS);
    for (const m of moves) assert.ok(isValid(twist(straight(), m.joint, m.direction)));
    assert.deepEqual(normalizeAngles([-1, 4, 5, 2]), [3, 0, 1, 2]);
    assert.deepEqual(twist([0, 0], 1, -1), [0, 3]);
});
