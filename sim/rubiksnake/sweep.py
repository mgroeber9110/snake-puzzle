"""Exact check of the path of a twist, a vectorized port of src/sweep.js.

Samples the rotation so that no vertex moves more than `max_step` between
samples and tests the real wedge prisms of the moving part against the fixed
part with the separating axis theorem, all samples and candidate pairs in one
batch. This is the `executor="kinematic"` fallback of the environment and the
reference the MuJoCo executor is compared with.
"""

from __future__ import annotations

import math

import numpy as np

from .kinematics import forward_kinematics, is_valid, twist

WEDGE_VERTS = np.array([
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
])
FACE_DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]], dtype=float)
EDGE_DIRS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, -1]], dtype=float)
RADIUS = math.sqrt(3) / 2
DEFAULT_TOLERANCE = 1e-4
DEFAULT_MAX_STEP = 0.05


def _axis_rotation(u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotation matrices (S, 3, 3) about unit axis u by angles theta (S,)."""
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=float)
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _intersect(ra, pa, rb, pb, tolerance):
    """Batched SAT for prisms with rotations (P,3,3) and centres (P,3)."""
    rta, rtb = ra.transpose(0, 2, 1), rb.transpose(0, 2, 1)
    va = WEDGE_VERTS @ rta + pa[:, None, :]                       # (P, 6, 3)
    vb = WEDGE_VERTS @ rtb + pb[:, None, :]
    fa, fb = FACE_DIRS @ rta, FACE_DIRS @ rtb                     # (P, 4, 3)
    ea, eb = EDGE_DIRS @ rta, EDGE_DIRS @ rtb
    cr = np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(len(ra), 16, 3)
    axes = np.concatenate([fa, fb, cr], axis=1)                   # (P, 24, 3)
    length = np.sqrt((axes * axes).sum(2))                        # (P, 24)
    at = axes.transpose(0, 2, 1)
    pa_ = va @ at                                                 # (P, 6, 24)
    pb_ = vb @ at
    overlap = np.minimum(pa_.max(1), pb_.max(1)) - np.maximum(pa_.min(1), pb_.min(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(length < 1e-9, np.inf, overlap / length)
    return (overlap > tolerance).all(axis=1)


def find_sweep_collision(angles, joint: int, direction: int, tolerance=DEFAULT_TOLERANCE,
                         max_step=DEFAULT_MAX_STEP):
    """First collision on the way of the twist, or None if it is free.

    Returns (fraction, (i, j)) like findSweepCollision in JS: the fraction of
    the twist at which the first overlap is found and a fixed/moving pair.
    """
    cells, rots = forward_kinematics(angles)
    cells = cells.astype(float)
    rots = rots.astype(float)
    pivot = cells[joint]
    moving = np.arange(joint + 1, len(cells))
    fixed = np.arange(0, joint + 1)
    reach = np.linalg.norm(cells[moving] - pivot, axis=1).max() + 2 * RADIUS
    steps = max(1, math.ceil(abs(direction) * math.pi / 2 * reach / max_step))
    frac = np.arange(1, steps + 1) / steps

    # Rotation of the moving part about the joint axis through the shared face.
    u = rots[joint + 1][:, 1]
    face = cells[joint] - 0.5 * rots[joint][:, 2]
    rot = _axis_rotation(u, direction * frac * math.pi / 2)        # (S, 3, 3)
    mc = (cells[moving] - face) @ rot.transpose(0, 2, 1) + face   # (S, M, 3)
    d2 = ((mc[:, :, None, :] - cells[fixed][None, None, :, :]) ** 2).sum(-1)
    s_idx, m_idx, f_idx = np.nonzero(d2 < 4 * RADIUS * RADIUS)
    if len(s_idx) == 0:
        return None
    ra = rots[fixed[f_idx]]
    pa = cells[fixed[f_idx]]
    rb = rot[s_idx] @ rots[moving[m_idx]]
    pb = mc[s_idx, m_idx]
    hit = _intersect(ra, pa, rb, pb, tolerance)
    if not hit.any():
        return None
    k = np.flatnonzero(hit)
    first = k[np.argmin(s_idx[k])]
    return float(frac[s_idx[first]]), (int(fixed[f_idx[first]]), int(moving[m_idx[first]]))


def is_twist_free(angles, joint: int, direction: int, **kw) -> bool:
    return is_valid(twist(angles, joint, direction)) and find_sweep_collision(angles, joint, direction, **kw) is None
