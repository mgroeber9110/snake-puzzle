"""MuJoCo model of the snake (section 3.2 of docs/training-environment.md).

Each wedge is a body with the exact prism as a convex mesh; body k+1 is a
child of body k, attached by a hinge on the shared square face. Wedge 0 is
fixed in the world (gravity is off for shape reaching, so only relative poses
matter). Every joint has a position actuator with a torque limit. All
physics values are made up and live in `Physics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .kinematics import JOINT_BASE
from .sweep import WEDGE_VERTS


@dataclass
class Physics:
    timestep: float = 0.002        # s
    wedge_mass: float = 0.001      # per wedge (cell edge = 1 length unit, all units made up)
    kp: float = 200.0              # position gain per rad
    kv: float = 10.0               # damping of the actuator per rad/s
    armature: float = 0.01         # reflected rotor inertia of each servo
    torque_limit: float = 40.0     # turns the longer half of a stick within about 0.5 s
    twist_time: float = 0.4        # s, the target ramps over this time
    time_limit: float = 0.8        # s, after this the move counts as blocked
    tolerance: float = 0.05        # rad, joint counts as arrived
    hold_stiffness: float = 1e5    # extra spring holding the joints that are not being twisted
    hold_damping: float = 80.0
    max_drift: float = 0.05        # rad; a move that pushes another joint further counts as blocked
    friction: float = 0.0          # sliding friction between wedges (0 = frictionless)
    shrink: float = 0.002          # every wedge face is moved inward by this much: avoids contacts of touching faces
    margin: float = 0.0
    solref: tuple = (0.005, 1.0)  # stiff contacts (time constant, damping ratio)


def _quat(m: np.ndarray) -> str:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, m.astype(float).ravel())
    return " ".join(f"{x:.12g}" for x in q)


def inset_wedge(clearance: float) -> np.ndarray:
    """Vertices of the wedge with every face moved inward by `clearance`.

    Scaling about the cell centre would not do: the sloped face passes through
    the centre and would stay where it is.
    """
    h = 0.5 - clearance
    lo = -h                                  # entry face y = -h and exit face z = -h
    top = -clearance * np.sqrt(2.0) + h      # slope y + z = -c*sqrt(2), at z = -h (and y = -h)
    return np.array([[-h, lo, top], [h, lo, top], [h, lo, lo],
                     [-h, lo, lo], [-h, top, lo], [h, top, lo]])


def build_mjcf(num_wedges: int = 24, physics: Physics | None = None) -> str:
    p = physics or Physics()
    verts = " ".join(f"{x:.6g}" for x in inset_wedge(p.shrink).ravel())
    condim = 3 if p.friction > 0 else 1
    base_quat = _quat(JOINT_BASE)
    body = []
    for k in range(num_wedges):
        indent = "  " * (k + 2)
        if k == 0:
            body.append(f'{indent}<body name="w0">')
        else:
            body.append(f'{indent}<body name="w{k}" pos="0 0 -1" quat="{base_quat}">')
            body.append(f'{indent}  <joint name="j{k - 1}" type="hinge" axis="0 1 0" '
                        f'armature="{p.armature}"/>')
        body.append(f'{indent}  <geom name="g{k}" type="mesh" mesh="wedge" mass="{p.wedge_mass}" '
                    f'condim="{condim}" solref="{p.solref[0]} {p.solref[1]}" friction="{p.friction} 0 0" margin="{p.margin}" '
                    f'rgba="{0.15 if k % 2 == 0 else 0.9} 0.4 0.8 1"/>')
    for k in reversed(range(num_wedges)):
        body.append("  " * (k + 2) + "</body>")
    acts = "\n".join(
        f'    <position name="a{j}" joint="j{j}" kp="{p.kp}" kv="{p.kv}" '
        f'forcerange="{-p.torque_limit} {p.torque_limit}" forcelimited="true"/>'
        for j in range(num_wedges - 1))
    return f"""<mujoco model="rubiksnake">
  <option timestep="{p.timestep}" gravity="0 0 0" integrator="implicitfast"/>
  <asset>
    <mesh name="wedge" vertex="{verts}"/>
  </asset>
  <worldbody>
{chr(10).join(body)}
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>
"""


def build_model(num_wedges: int = 24, physics: Physics | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_mjcf(num_wedges, physics))
