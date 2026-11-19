"""Target shapes and the three curriculum stages (section 4.4 of the draft)."""

from __future__ import annotations

import numpy as np

from .kinematics import forward_kinematics, straight, twist, valid_move_mask
from .notation import parse_notation

# Wolter notation, same strings as test/notation.test.js and scripts/export-vectors.js.
NAMED = {
    "wolf": "2R2-3L2-4L2-5R2-6R2-7L2-8L2-10L2-10R2-12R2",
    "cat": "9R2-9L2-8L2-7R2-6R2-6L2-5L3-4L2-3R2-2R2-2L2",
    "snowflake": "1R3-2L1-2R3-3L3-3R1-4L3-4R1-5L1-5R3-6L1-6R3-7L3-7R1-8L3-8R1-9L1-9R3-10L1-10R3-12L3-11R1-11L3-12R1",
    "cobra": "2L3-2R3-3L1-3R1-4L1-5L3-4R1-5R3-6L3-7L1-6R3-7R3-8L1-9L1-9R1-11L2-12R1",
    "ball": "1R1-2L3-2R3-3L1-3R1-4L1-4R1-5L3-5R3-12R3-12L3-11R3-11L3-10R1-10L1-9R1-9L1-8R3-8L3-7R1-6R3-6L3-7L1",
}


def named(name: str) -> np.ndarray:
    return parse_notation(NAMED[name])


def random_walk(rng: np.random.Generator, start, steps: int, executor=None) -> np.ndarray:
    """`steps` random moves from `start`, each with a valid end state and, if an
    executor is given, completed by it. Returns early if every move is blocked."""
    angles = np.array(start, dtype=np.int64)
    for _ in range(steps):
        actions = np.flatnonzero(valid_move_mask(angles))
        rng.shuffle(actions)
        for a in actions:
            joint, direction = int(a // 2), 1 - 2 * int(a % 2)
            if executor is None or executor.execute(angles, joint, direction).done:
                angles = twist(angles, joint, direction)
                break
        else:
            break
    return angles


def near_shape(rng, depth: int, num_wedges: int = 24, executor=None) -> np.ndarray:
    """Stage 1: a configuration `depth` random moves from the straight stick."""
    return random_walk(rng, straight(num_wedges), depth, executor)


def flat_walk(rng, half_turns: int, num_wedges: int = 24, executor=None) -> np.ndarray:
    """Stage 2: a flat shape made by `half_turns` random half turns from the stick.

    Each half turn is two quarter turns in the same direction, both with a
    valid end state and completed by the executor, so the shape is reachable
    by construction. A joint already at a half turn may be turned back.
    """
    angles = straight(num_wedges)
    n = num_wedges - 1
    for _ in range(half_turns):
        for _try in range(4 * n):
            joint = int(rng.integers(n))
            direction = 1 if rng.random() < 0.5 else -1
            a = angles
            ok = True
            for _q in range(2):
                if not valid_move_mask(a)[2 * joint + (0 if direction == 1 else 1)] or \
                        (executor is not None and not executor.execute(a, joint, direction).done):
                    ok = False
                    break
                a = twist(a, joint, direction)
            if ok:
                angles = a
                break
    return angles


def gyration(angles) -> float:
    """Radius of gyration of the occupied cells (small = compact)."""
    cells, _ = forward_kinematics(angles)
    c = cells - cells.mean(0)
    return float(np.sqrt((c * c).sum(1).mean()))


def compact_walk(rng, executor=None, steps: int = 80, beta: float = 3.0, num_wedges: int = 24):
    """Compact 3D shapes: a random walk from the stick that prefers moves making
    the shape more compact; returns the most compact state after at least 20 moves.
    Every move has a valid end state and, with an executor, is completed by it."""
    a = straight(num_wedges)
    best, best_rg = a.copy(), np.inf
    for k in range(steps):
        acts = np.flatnonzero(valid_move_mask(a))
        if len(acts) == 0:
            break
        cand = [(int(x // 2), 1 - 2 * int(x % 2)) for x in acts]
        rg = np.array([gyration(twist(a, j, d)) for j, d in cand])
        p = np.exp(-beta * (rg - rg.min()))
        order = rng.choice(len(cand), size=len(cand), replace=False, p=p / p.sum())
        for i in order:
            j, d = cand[i]
            if executor is None or executor.execute(a, j, d).done:
                a = twist(a, j, d)
                break
        else:
            break
        if k >= 20 and gyration(a) < best_rg:
            best, best_rg = a.copy(), gyration(a)
    return best


def is_flat_in_layer(angles) -> bool:
    """All wedge cells in one lattice layer (like the straight stick)."""
    cells, _ = forward_kinematics(angles)
    return any(len(set(cells[:, k].tolist())) == 1 for k in range(3))


def random_flat_shape(rng, num_wedges: int = 24, turned: int = 6, tries: int = 10_000):
    """Stage 2: random half-turn configuration that stays in one lattice layer."""
    from .kinematics import is_valid
    n = num_wedges - 1
    for _ in range(tries):
        angles = np.zeros(n, dtype=np.int64)
        angles[rng.choice(n, size=turned, replace=False)] = 2
        if is_valid(angles) and is_flat_in_layer(angles):
            return angles
    raise RuntimeError("no flat shape found")


STAGES = {
    1: "near shapes, a few moves from the straight stick",
    2: "flat shapes: generated one-layer half-turn shapes, Wolf, then Snowflake",
    3: "3D shapes: Cobra, then Ball",
}
