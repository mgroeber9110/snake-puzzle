// Collision check for the path of a twist, not just its end state.
//
// kinematics.js decides exactly whether an integer configuration is valid.
// During a twist the moving part of the snake passes through non-lattice
// poses, so this module tests the actual wedge geometry (triangular prisms)
// with the separating axis theorem at finely sampled intermediate angles.
// No rendering dependencies; runs in the browser and in Node.

import { forwardKinematics, twist, isValid } from './kinematics.js';

// Wedge in its local frame (unit cube cell, see kinematics.js):
// { |x|,|y|,|z| <= 1/2, y + z <= 0 }.
const WEDGE_VERTS = [
    [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
];

// Local directions for the separating axis test of two such prisms.
// Face normals: x (triangles), y (entry face), z (exit face), (0,1,1) (slope).
// Edge directions: x, y, z, (0,1,-1) (slope edges of the triangles).
const FACE_DIRS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]];
const EDGE_DIRS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, -1]];

// Penetration depth below which two wedges count as merely touching. Faces of
// wedges in valid configurations coincide exactly.
const DEFAULT_TOLERANCE = 1e-4;

// Largest distance any vertex may move between two samples of a twist.
const DEFAULT_MAX_STEP = 0.05;

function rotate(r, v) {
    return [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ];
}

function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** World-space geometry of a wedge pose from forwardKinematics(). */
export function wedgeShape(pose) {
    const { position: p, rotation: r } = pose;
    return {
        center: p,
        verts: WEDGE_VERTS.map(v => {
            const w = rotate(r, v);
            return [w[0] + p[0], w[1] + p[1], w[2] + p[2]];
        }),
        faces: FACE_DIRS.map(d => rotate(r, d)),
        edges: EDGE_DIRS.map(d => rotate(r, d)),
    };
}

// Overlap of both projections onto axis, normalized by the axis length.
function overlapOnAxis(axis, a, b) {
    const len = Math.hypot(axis[0], axis[1], axis[2]);
    if (len < 1e-9) return Infinity; // parallel edges give no usable axis
    let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
    for (const v of a.verts) {
        const p = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
        if (p < minA) minA = p;
        if (p > maxA) maxA = p;
    }
    for (const v of b.verts) {
        const p = v[0] * axis[0] + v[1] * axis[1] + v[2] * axis[2];
        if (p < minB) minB = p;
        if (p > maxB) maxB = p;
    }
    return (Math.min(maxA, maxB) - Math.max(minA, minB)) / len;
}

// Circumradius of the unit cube around a wedge's cell centre.
const RADIUS = Math.sqrt(3) / 2;

/**
 * Exact intersection test of two wedge shapes. Shapes that only touch
 * (penetration <= tolerance) do not intersect.
 */
export function shapesIntersect(a, b, tolerance = DEFAULT_TOLERANCE) {
    const dx = a.center[0] - b.center[0];
    const dy = a.center[1] - b.center[1];
    const dz = a.center[2] - b.center[2];
    if (dx * dx + dy * dy + dz * dz >= 4 * RADIUS * RADIUS) return false;

    for (const axis of a.faces) if (overlapOnAxis(axis, a, b) <= tolerance) return false;
    for (const axis of b.faces) if (overlapOnAxis(axis, a, b) <= tolerance) return false;
    for (const ea of a.edges) {
        for (const eb of b.edges) {
            if (overlapOnAxis(cross(ea, eb), a, b) <= tolerance) return false;
        }
    }
    return true;
}

/**
 * All pairs [i, j] of overlapping wedges for arbitrary (also fractional)
 * angles in quarter turns. Directly connected wedges are skipped: they share
 * a square face and turn about its normal, so they never penetrate.
 * @param {(i: number, j: number) => boolean} [pairFilter] restricts the tested pairs
 */
export function findGeometricCollisions(angles, { tolerance = DEFAULT_TOLERANCE, pairFilter } = {}) {
    const shapes = forwardKinematics(angles).map(wedgeShape);
    const collisions = [];
    for (let i = 0; i < shapes.length; i++) {
        for (let j = i + 2; j < shapes.length; j++) {
            if (pairFilter && !pairFilter(i, j)) continue;
            if (shapesIntersect(shapes[i], shapes[j], tolerance)) collisions.push([i, j]);
        }
    }
    return collisions;
}

/**
 * First collision on the way of twisting `joint` by `direction` quarter turns
 * from the integer configuration `angles`, or null if the whole twist is free.
 * The angle is sampled so finely that no vertex moves more than maxStep
 * between two samples. Only wedges on opposite sides of the joint are tested;
 * wedges on the same side keep their relative pose.
 *
 * @returns {null | {fraction: number, pair: [number, number]}}
 *          fraction of the twist (0..1] at which the first overlap was found
 */
export function findSweepCollision(angles, joint, direction,
                                   { tolerance = DEFAULT_TOLERANCE, maxStep = DEFAULT_MAX_STEP } = {}) {
    const pairFilter = (i, j) => (i <= joint) !== (j <= joint);

    // No point moves farther from the joint axis than the farthest moving
    // wedge centre is from the joint's cell, plus one cube diagonal.
    const poses = forwardKinematics(angles);
    const pivot = poses[joint].position;
    let reach = 0;
    for (let i = joint + 1; i < poses.length; i++) {
        const p = poses[i].position;
        reach = Math.max(reach, Math.hypot(p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]));
    }
    reach += 2 * RADIUS;
    const arc = Math.abs(direction) * Math.PI / 2 * reach;
    const steps = Math.max(1, Math.ceil(arc / maxStep));

    const frame = angles.slice();
    for (let k = 1; k <= steps; k++) {
        const fraction = k / steps;
        frame[joint] = angles[joint] + direction * fraction;
        const hits = findGeometricCollisions(frame, { tolerance, pairFilter });
        if (hits.length > 0) return { fraction, pair: hits[0] };
    }
    return null;
}

/** True if the twist is possible: valid end state and nothing blocks the way. */
export function isTwistFree(angles, joint, direction, options) {
    return isValid(twist(angles, joint, direction)) &&
        findSweepCollision(angles, joint, direction, options) === null;
}
