// Pure kinematic model of the Rubik's Snake. No rendering dependencies, so it
// runs unchanged in the browser and in Node (tests, training).
//
// Model
// -----
// The snake is a chain of N wedges (24 for the standard puzzle) connected by
// N-1 joints. Every wedge is half of a unit cube, cut along the diagonal plane
// through two opposite edges. In its local frame a wedge is centred at the
// origin and occupies the half cube { |x|,|y|,|z| <= 1/2, y + z <= 0 }. It has
// two full square faces:
//   - the entry face at y = -1/2 (local -Y), facing the previous wedge,
//   - the exit face  at z = -1/2 (local -Z), facing the next wedge.
//
// Joint j connects wedge j and wedge j+1 and twists about the normal of their
// shared square face. Joint angles are given in quarter turns (0..3); angle 0
// everywhere is the straight stick. Fractional angles are accepted by the
// forward kinematics for animation, collision checks require integers.
//
// Because every face is axis aligned, every wedge of an integer configuration
// sits in a cell of the integer lattice. Two wedges can only overlap when they
// share a cell, and two half cubes in the same cell are disjoint exactly when
// they are complementary (opposite entry and exit faces). This makes the
// collision check exact.

export const NUM_WEDGES = 24;
export const NUM_JOINTS = NUM_WEDGES - 1;

// 3x3 matrices are row-major arrays of 9 numbers.
const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// Fixed rotation from wedge j's frame to wedge j+1's frame at angle 0:
// X -> -X, Y -> -Z, Z -> -Y (columns are the images of X, Y, Z). It maps the
// next wedge's entry face (-Y) onto the previous wedge's exit face direction
// reversed (+Z), and the next exit face (-Z) onto +Y, giving the straight stick.
const JOINT_BASE = [
    -1, 0, 0,
    0, 0, -1,
    0, -1, 0,
];

function matMul(a, b) {
    const r = new Array(9);
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j] + 0; // + 0 turns -0 into 0
        }
    }
    return r;
}

function matVec(m, v) {
    return [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + 0,
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2] + 0,
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2] + 0,
    ];
}

// cos/sin of a quarter-turn count, exact for integers.
function quarterCosSin(q) {
    if (Number.isInteger(q)) {
        const k = ((q % 4) + 4) % 4;
        return [[1, 0], [0, 1], [-1, 0], [0, -1]][k];
    }
    const a = q * Math.PI / 2;
    return [Math.cos(a), Math.sin(a)];
}

// Rotation about the local Y axis (the joint axis) by q quarter turns.
function rotY(q) {
    const [c, s] = quarterCosSin(q);
    return [
        c, 0, s,
        0, 1, 0,
        -s, 0, c,
    ];
}

/**
 * Forward kinematics. Returns one pose per wedge: `position` is the centre of
 * the wedge's cube cell, `rotation` the row-major 3x3 matrix from the wedge's
 * local frame to the world frame. Wedge 0 sits at the origin with identity
 * rotation. Integer angles yield exact integer poses.
 *
 * @param {number[]} angles joint angles in quarter turns, length N-1
 */
export function forwardKinematics(angles) {
    let rotation = IDENTITY;
    let position = [0, 0, 0];
    const poses = [{ position, rotation }];
    for (let j = 0; j < angles.length; j++) {
        // Step through the exit face (-Z) of the current wedge.
        const exit = matVec(rotation, [0, 0, -1]);
        position = [position[0] + exit[0], position[1] + exit[1], position[2] + exit[2]];
        rotation = matMul(matMul(rotation, JOINT_BASE), rotY(angles[j]));
        poses.push({ position, rotation });
    }
    return poses;
}

/** World direction of a wedge's entry face (local -Y). */
export function entryDirection(pose) {
    const r = pose.rotation;
    return [0 - r[1], 0 - r[4], 0 - r[7]];
}

/** World direction of a wedge's exit face (local -Z). */
export function exitDirection(pose) {
    const r = pose.rotation;
    return [0 - r[2], 0 - r[5], 0 - r[8]];
}

/** Reduce angles to 0..3. */
export function normalizeAngles(angles) {
    return angles.map(a => ((Math.round(a) % 4) + 4) % 4);
}

function assertIntegerAngles(angles) {
    for (const a of angles) {
        if (!Number.isInteger(a)) throw new Error(`collision check needs integer quarter turns, got ${a}`);
    }
}

function key(v) {
    return `${v[0]},${v[1]},${v[2]}`;
}

function same(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function opposite(a, b) {
    return a[0] === -b[0] && a[1] === -b[1] && a[2] === -b[2];
}

// Two half cubes in the same cell are disjoint iff their square faces are
// opposite: {entryA, exitA} == {-entryB, -exitB}.
function complementary(pa, pb) {
    const ea = entryDirection(pa), xa = exitDirection(pa);
    const eb = entryDirection(pb), xb = exitDirection(pb);
    return (opposite(ea, eb) && opposite(xa, xb)) || (opposite(ea, xb) && opposite(xa, eb));
}

/**
 * All pairs [i, j] (i < j) of wedges whose interiors overlap.
 * @param {number[]} angles integer joint angles
 */
export function findCollisions(angles) {
    assertIntegerAngles(angles);
    const poses = forwardKinematics(angles);
    const cells = new Map();
    const collisions = [];
    poses.forEach((pose, i) => {
        const k = key(pose.position);
        const others = cells.get(k);
        if (!others) {
            cells.set(k, [i]);
            return;
        }
        for (const o of others) {
            if (!complementary(poses[o], pose)) collisions.push([o, i]);
        }
        others.push(i);
    });
    return collisions.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

/** True if no two wedges overlap. */
export function isValid(angles) {
    return findCollisions(angles).length === 0;
}

/** New angle array with joint `joint` twisted by `direction` quarter turns. */
export function twist(angles, joint, direction) {
    const next = angles.slice();
    next[joint] = (((next[joint] + direction) % 4) + 4) % 4;
    return next;
}

/**
 * All single quarter-turn moves leading to a valid configuration.
 * Only the end state is checked, not the sweep during the twist.
 * @returns {{joint: number, direction: number}[]}
 */
export function validMoves(angles) {
    const moves = [];
    for (let joint = 0; joint < angles.length; joint++) {
        for (const direction of [1, -1]) {
            if (isValid(twist(angles, joint, direction))) moves.push({ joint, direction });
        }
    }
    return moves;
}

/** Straight stick: all joint angles 0. */
export function straight(numWedges = NUM_WEDGES) {
    return new Array(numWedges - 1).fill(0);
}
