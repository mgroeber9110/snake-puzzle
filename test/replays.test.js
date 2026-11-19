import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { isValid, twist, normalizeAngles } from '../src/kinematics.js';
import { isTwistFree } from '../src/sweep.js';
import { parseNotation } from '../src/notation.js';

// The example trajectories offered by snake4.html carry their target as Wolter
// notation, and replaying them with the exact checks gives what meta.success says.
const dir = new URL('../src/replays/', import.meta.url);
const index = JSON.parse(readFileSync(new URL('index.json', dir), 'utf8'));

for (const { file } of index) {
    test(`replay ${file}`, () => {
        const t = JSON.parse(readFileSync(new URL(file, dir), 'utf8'));
        assert.equal(typeof t.meta.target, 'string', 'meta.target must be Wolter notation');
        const target = parseNotation(t.meta.target);
        assert.ok(isValid(target));
        let angles = normalizeAngles(t.start);
        for (const [joint, direction] of t.moves) {
            if (isValid(twist(angles, joint, direction)) && isTwistFree(angles, joint, direction)) {
                angles = twist(angles, joint, direction);
            }
        }
        const reached = angles.join() === target.join() || angles.join() === target.slice().reverse().join();
        assert.equal(reached, t.meta.success);
    });
}
