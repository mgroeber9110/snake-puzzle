"""Wolter shape notation, a port of src/notation.js (see there for the mapping).

Dark wedge n is model wedge 2(n-1); nR is joint 2n-2, nL is joint 2n-3;
position p is angle p on a left joint and 4-p on a right joint.
"""

from __future__ import annotations

import re

import numpy as np

from .kinematics import NUM_WEDGES

_TOKEN = re.compile(r"^(\d+)([LR])([0-3])$", re.IGNORECASE)


def _to_angle(position: int, side: str) -> int:
    return (4 - position) % 4 if side == "R" else position


def joint_of(piece: int, side: str) -> int:
    return 2 * piece - 2 if side == "R" else 2 * piece - 3


def piece_of(joint: int) -> tuple[int, str]:
    return (joint // 2 + 1, "R") if joint % 2 == 0 else ((joint + 3) // 2, "L")


def parse_notation(text: str, num_wedges: int = NUM_WEDGES) -> np.ndarray:
    if num_wedges % 2:
        raise ValueError(f"notation needs an even number of wedges, got {num_wedges}")
    pieces = num_wedges // 2
    angles = np.zeros(num_wedges - 1, dtype=np.int64)
    seen = set()
    for token in [t for t in re.split(r"[\s,\-]+", text.strip()) if t]:
        m = _TOKEN.match(token)
        if not m:
            raise ValueError(f'invalid turn "{token}", expected e.g. "11L2"')
        piece, side, pos = int(m.group(1)), m.group(2).upper(), int(m.group(3))
        if not 1 <= piece <= pieces:
            raise ValueError(f'invalid turn "{token}": wedge must be 1..{pieces}')
        if piece == 1 and side == "L":
            raise ValueError(f'invalid turn "{token}": wedge 1 has no left joint')
        joint = joint_of(piece, side)
        if joint in seen:
            raise ValueError(f'invalid turn "{token}": {piece}{side} given twice')
        seen.add(joint)
        angles[joint] = _to_angle(pos, side)
    return angles


def to_notation(angles) -> str:
    if (len(angles) + 1) % 2:
        raise ValueError(f"notation needs an even number of wedges, got {len(angles) + 1}")
    turns = []
    for joint, a in enumerate(angles):
        piece, side = piece_of(joint)
        pos = _to_angle(int(a) % 4, side)
        if pos:
            turns.append(f"{piece}{side}{pos}")
    return "-".join(turns)
