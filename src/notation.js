// Shape notation by Thomas Wolter (http://thomas-wolter.de/rubik_notation_en.html),
// mapped onto the joint angles of the kinematic model in kinematics.js.
//
// Notation
// --------
// In the base position (straight stick) the 12 dark wedges are numbered 1..12
// from left to right; the leftmost wedge is dark, the rightmost bright. Every
// dark wedge has a bright neighbour on its right ("R") and, except wedge 1,
// one on its left ("L"). A turn is written as <dark wedge><side><position>,
// e.g. "11L2": the bright wedge left of dark wedge 11 is turned to position 2.
// Positions 1, 2, 3 are the three quarter-turn positions, the base position
// gets no number and joints in base position are left out. A shape is the
// list of its turns joined by "-", e.g. the Wolf "2R2-3L2-4L2-...".
//
// Mapping onto the model
// ----------------------
// Wedge 0 of the model is the leftmost dark wedge, so dark wedge n is model
// wedge 2(n-1), and
//   nR -> joint 2n-2 (between wedge 2n-2 and 2n-1),
//   nL -> joint 2n-3 (between wedge 2n-3 and 2n-2).
// Position p of a left joint is joint angle p, position p of a right joint is
// joint angle 4 - p (mod 4): the bright wedges on both sides of a dark wedge
// turn in the same sense as seen along the stick, which is the opposite sense
// as seen from the dark wedge. This choice is fixed by two checks:
//   - left vs. right: only with opposite signs does the Ball example become the
//     ball (3x3x3 block) and the Snowflake a flat hexagonal ring; otherwise
//     the two come out swapped.
//   - the overall sign (which of two mirror images is meant): the picture of
//     the Cobra (2L3-2R3-3L1-...-12R1) on the "fans & friends" page of the same
//     site only matches this choice; the other one gives its mirror image.

import { NUM_WEDGES } from './kinematics.js';

// Angle for a position on side 'L' or 'R'; the mapping is its own inverse.
function toAngle(position, side) {
    return side === 'R' ? (4 - position) % 4 : position;
}

function toPosition(angle, side) {
    return toAngle(angle, side);
}

/** Joint index for dark wedge `piece` (1-based) and side 'L' or 'R'. */
export function jointOf(piece, side) {
    return side === 'R' ? 2 * piece - 2 : 2 * piece - 3;
}

/** Inverse of jointOf: { piece, side } for a joint index. */
export function pieceOf(joint) {
    return joint % 2 === 0
        ? { piece: joint / 2 + 1, side: 'R' }
        : { piece: (joint + 3) / 2, side: 'L' };
}

/**
 * Parse a shape in Wolter notation into joint angles (quarter turns 0..3).
 * Turns may be separated by "-", ",", or whitespace and in any order; letters
 * are case insensitive. Position 0 (base position) is accepted. An empty
 * string is the straight stick. Throws on malformed or repeated turns.
 *
 * @param {string} text e.g. "2R2-3L2-4L2-5R2"
 * @param {number} numWedges even number of wedges (24 for the standard snake)
 * @returns {number[]} joint angles, length numWedges - 1
 */
export function parseNotation(text, numWedges = NUM_WEDGES) {
    if (numWedges % 2 !== 0) throw new Error(`notation needs an even number of wedges, got ${numWedges}`);
    const pieces = numWedges / 2;
    const angles = new Array(numWedges - 1).fill(0);
    const seen = new Set();
    const tokens = text.trim().split(/[\s,\-]+/).filter(t => t.length > 0);
    for (const token of tokens) {
        const m = /^(\d+)([LR])([0-3])$/i.exec(token);
        if (!m) throw new Error(`invalid turn "${token}", expected e.g. "11L2"`);
        const piece = Number(m[1]);
        const side = m[2].toUpperCase();
        if (piece < 1 || piece > pieces) throw new Error(`invalid turn "${token}": wedge must be 1..${pieces}`);
        if (piece === 1 && side === 'L') throw new Error(`invalid turn "${token}": wedge 1 has no left joint`);
        const joint = jointOf(piece, side);
        if (seen.has(joint)) throw new Error(`invalid turn "${token}": ${piece}${side} given twice`);
        seen.add(joint);
        angles[joint] = toAngle(Number(m[3]), side);
    }
    return angles;
}

/**
 * Write joint angles in Wolter notation: turns ordered by dark wedge, left
 * before right, base positions left out. The straight stick gives "".
 *
 * @param {number[]} angles integer joint angles
 */
export function toNotation(angles) {
    if ((angles.length + 1) % 2 !== 0) throw new Error(`notation needs an even number of wedges, got ${angles.length + 1}`);
    const turns = [];
    // Joint order 0, 1, 2, ... is 1R, 2L, 2R, 3L, ... which is already
    // ordered by dark wedge with left before right.
    angles.forEach((a, joint) => {
        if (!Number.isInteger(a)) throw new Error(`notation needs integer quarter turns, got ${a}`);
        const { piece, side } = pieceOf(joint);
        const position = toPosition(((a % 4) + 4) % 4, side);
        if (position === 0) return;
        turns.push(`${piece}${side}${position}`);
    });
    return turns.join('-');
}
