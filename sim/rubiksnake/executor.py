"""Executors carry out a discrete twist and report whether it got through.

`MujocoExecutor` runs the twist with simulated servos (section 3.2): the
target of the joint ramps by 90 degrees, the simulation runs until the joint
arrives or the time limit is hit; a stall, or a twist that pushes another
joint off by more than `max_drift`, counts as blocked.
`KinematicExecutor` uses the exact sweep check (sweep.py). Both return a
`TwistResult`; neither changes the integer state, the environment does.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .kinematics import is_valid, twist
from .sweep import find_sweep_collision


@dataclass
class TwistResult:
    done: bool          # the twist completed; the new state is twist(angles, joint, direction)
    blocked: bool       # stopped by a collision on the way
    sim_time: float     # simulated seconds the move took (0 for the kinematic executor)


class KinematicExecutor:
    """Exact geometric check, cached on (angles, joint, direction)."""

    def __init__(self, cache_size: int = 200_000):
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size

    def execute(self, angles, joint: int, direction: int) -> TwistResult:
        key = (bytes(np.asarray(angles, dtype=np.int8)), joint, direction)
        free = self.cache.get(key)
        if free is None:
            free = is_valid(twist(angles, joint, direction)) and \
                find_sweep_collision(angles, joint, direction) is None
            self.cache[key] = free
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return TwistResult(done=free, blocked=not free, sim_time=0.0)


class MujocoExecutor:
    def __init__(self, num_wedges: int = 24, physics=None, cache_size: int = 200_000):
        import mujoco

        from .mjcf import Physics, build_model

        self.mujoco = mujoco
        self.physics = physics or Physics()
        self.model = build_model(num_wedges, self.physics)
        self.data = mujoco.MjData(self.model)
        self.num_joints = num_wedges - 1
        self.max_steps = int(round(self.physics.time_limit / self.physics.timestep))
        self.contype0 = self.model.geom_contype.copy()
        self.cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self.conaff0 = self.model.geom_conaffinity.copy()

    def _set_state(self, angles):
        q = np.asarray(angles, dtype=float) * (math.pi / 2)
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        self.data.ctrl[:] = q
        self.data.act[:] = 0 if self.model.na else self.data.act
        self.data.time = 0
        self.mujoco.mj_forward(self.model, self.data)

    def execute(self, angles, joint: int, direction: int, record=None) -> TwistResult:
        """Run the twist; results are cached since the physics is deterministic."""
        key = (bytes(np.asarray(angles, dtype=np.int8)), joint, direction)
        if record is None and key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = self._simulate(angles, joint, direction, record)
        self.cache[key] = result
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def _simulate(self, angles, joint, direction, record):
        m, d, p, mj = self.model, self.data, self.physics, self.mujoco
        # Only wedges on opposite sides of the joint can collide during the twist.
        # Geom k belongs to wedge k; wedges 0..joint stay, joint+1.. move.
        m.geom_contype[: joint + 1] = 0
        m.geom_conaffinity[: joint + 1] = 1
        m.geom_contype[joint + 1:] = 1
        m.geom_conaffinity[joint + 1:] = 0
        self._set_state(angles)
        start = d.qpos[joint]
        goal = start + direction * math.pi / 2
        ramp = max(1, int(round(p.twist_time / p.timestep)))
        others = np.ones(self.num_joints, dtype=bool)
        others[joint] = False
        q_other = d.qpos[others].copy()
        steps_done = self.max_steps
        arrived = False
        drift = 0.0
        for k in range(1, self.max_steps + 1):
            d.ctrl[joint] = start + (goal - start) * min(1.0, k / ramp)
            mj.mj_step(m, d)
            if k % 5 == 0:
                drift = max(drift, np.abs(d.qpos[others] - q_other).max(initial=0.0))
                if drift > p.max_drift:
                    break  # the twist only gets through by bending the rest of the snake
            if record is not None and k % 10 == 0:
                record.append(d.qpos.copy())
            if k >= ramp and abs(d.qpos[joint] - goal) < p.tolerance and abs(d.qvel[joint]) < 1.0:
                arrived = True
                steps_done = k
                break
        # Other joints must stay put: the servos are stiff and have no play, so
        # a twist that only gets through by bending the rest of the snake is blocked.
        drift = max(drift, np.abs(d.qpos[others] - q_other).max(initial=0.0))
        done = arrived and drift <= p.max_drift
        if done and not is_valid(twist(angles, joint, direction)):
            done = False  # safety net, should not happen with masked actions
        m.geom_contype[:] = self.contype0
        m.geom_conaffinity[:] = self.conaff0
        return TwistResult(done=bool(done), blocked=not done, sim_time=steps_done * p.timestep)
