"""A* baseline: shortest move sequence between two configurations.

The heuristic is the cyclic angle distance (each quarter turn changes it by at
most 1, so it is admissible). Moves must have a valid end state and, with an
executor, be completed by it. A target also counts as reached in the reversed
reading (the same shape read from the other end is the reversed angle list).
"""

from __future__ import annotations

import heapq
import itertools

import numpy as np

from .kinematics import twist, valid_move_mask


def _dist(a: np.ndarray, targets) -> int:
    best = None
    for t in targets:
        d = np.abs(a - t) % 4
        v = int(np.minimum(d, 4 - d).sum())
        best = v if best is None else min(best, v)
    return best


def astar(start, target, executor=None, max_expansions: int = 200_000, reversed_ok: bool = True):
    """Returns a list of (joint, direction) moves, or None if not found within the budget.

    Edges are checked lazily: the executor only runs for a move when the
    state it leads to is popped from the queue, which keeps the number of
    simulated twists close to the number of expanded states.
    """
    start = np.asarray(start, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    targets = [target, target[::-1].copy()] if reversed_ok else [target]
    goal_keys = {t.tobytes() for t in targets}
    counter = itertools.count()
    # Entries: (f, tie, g, state, parent_key, move, parent_state)
    open_ = [(_dist(start, targets), next(counter), 0, start, None, None, None)]
    parent: dict = {}
    expansions = 0
    while open_:
        _, _, g, s, pkey, move, pstate = heapq.heappop(open_)
        key = s.tobytes()
        if key in parent:
            continue
        if move is not None and executor is not None and \
                not executor.execute(pstate, move[0], move[1]).done:
            continue
        parent[key] = (pkey, move)
        if key in goal_keys:
            moves = []
            while parent[key][1] is not None:
                key, m = parent[key][0], parent[key][1]
                moves.append(m)
            return moves[::-1]
        expansions += 1
        if expansions > max_expansions:
            return None
        for a in np.flatnonzero(valid_move_mask(s)):
            joint, direction = int(a // 2), 1 - 2 * int(a % 2)
            nxt = twist(s, joint, direction)
            if nxt.tobytes() in parent:
                continue
            heapq.heappush(open_, (g + 1 + _dist(nxt, targets), next(counter), g + 1, nxt,
                                   key, (joint, direction), s))
    return None
