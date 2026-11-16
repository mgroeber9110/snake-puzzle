"""Gymnasium environment for the Rubik's Snake (section 4 of the draft).

Task `reach_shape`: twist the joints until the snake has the target shape.
Actions are Discrete(2 * joints): action a twists joint a // 2 by +1 (a even)
or -1 (a odd) quarter turn, the (joint, direction) pairs of twist().
"""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .kinematics import forward_kinematics, straight, twist, valid_move_mask
from .notation import parse_notation
from .shapes import near_shape, random_walk

STEP_PENALTY = -0.01
SUCCESS_REWARD = 1.0
BLOCKED_PENALTY = -0.05
INVALID_PENALTY = -1.0


def make_executor(kind: str, num_wedges: int, physics=None):
    if kind == "mujoco":
        from .executor import MujocoExecutor
        return MujocoExecutor(num_wedges, physics)
    if kind == "kinematic":
        from .executor import KinematicExecutor
        return KinematicExecutor()
    if kind == "none":  # end-state check only, nothing blocks on the way
        return None
    raise ValueError(f"unknown executor {kind!r}")


class RubikSnakeEnv(gym.Env):
    """Goal-conditioned shape reaching.

    targets: None for stage-1 near shapes (`target_depth` random moves from the
        stick), or a sequence of angle arrays / Wolter strings, or a callable
        rng -> angles.
    curriculum_depth: if set, an episode starts this many random completed
        moves away from the target (reverse curriculum); otherwise at `start`
        (default the straight stick).
    """

    metadata = {"render_modes": []}

    def __init__(self, num_wedges: int = 24, task: str = "reach_shape",
                 targets: Sequence | Callable | None = None, executor: str = "mujoco",
                 physics=None, max_steps: int = 200, curriculum_depth: int | None = None,
                 target_depth: int | tuple[int, int] = 4, start=None, gamma: float = 0.99,
                 geometry: bool = False, delta: bool = True):
        if task != "reach_shape":
            raise NotImplementedError("only task='reach_shape' is implemented so far")
        self.num_wedges = num_wedges
        self.num_joints = num_wedges - 1
        self.executor = make_executor(executor, num_wedges, physics)
        self.max_steps_default = max_steps
        self.curriculum_depth = curriculum_depth
        self.target_depth = target_depth
        self.start = None if start is None else np.asarray(start, dtype=np.int64)
        self.gamma = gamma
        self.geometry = geometry
        self.delta = delta
        if targets is None or callable(targets):
            self.targets = targets
        else:
            self.targets = [parse_notation(t, num_wedges) if isinstance(t, str)
                            else np.asarray(t, dtype=np.int64) for t in targets]

        n = self.num_joints
        obs = {
            "angles": spaces.MultiDiscrete([4] * n),
            "target": spaces.MultiDiscrete([4] * n),
            "action_mask": spaces.MultiBinary(2 * n),
        }
        # Moves tried in the current state and blocked on the way; cleared
        # when the state changes. Without it a policy without memory would
        # repeat a blocked move forever.
        obs["blocked"] = spaces.MultiBinary(2 * n)
        if delta:
            # (target - angles) mod 4: redundant, but saves the network from
            # learning 23 comparisons of one-hot codes.
            obs["delta"] = spaces.MultiDiscrete([4] * n)
        if geometry:
            obs["geometry"] = spaces.Box(-num_wedges, num_wedges, (num_wedges, 12), np.float32)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Discrete(2 * n)
        self.angles = straight(num_wedges)
        self.target = straight(num_wedges)

    # -- curriculum hooks (called from training callbacks via env_method) --
    def set_target_depth(self, depth):
        self.target_depth = depth

    def set_max_steps(self, max_steps):
        self.max_steps_default = max_steps

    def set_curriculum_depth(self, depth):
        self.curriculum_depth = depth

    # -- helpers --
    def distance(self, angles=None) -> int:
        a = self.angles if angles is None else angles
        best = None
        for t in (self.target, self.target[::-1]):
            d = np.abs(a - t) % 4
            v = int(np.minimum(d, 4 - d).sum())
            best = v if best is None else min(best, v)
        return best

    def is_success(self, angles=None) -> bool:
        a = self.angles if angles is None else angles
        return bool(np.array_equal(a, self.target) or np.array_equal(a, self.target[::-1]))

    def _potential(self, angles) -> float:
        return -self.distance(angles) / self.num_joints

    def action_masks(self) -> np.ndarray:
        return self._mask

    def _obs(self):
        o = {"angles": self.angles.copy(), "target": self.target.copy(),
             "action_mask": self._mask.astype(np.int8), "blocked": self._blocked.astype(np.int8)}
        if self.delta:
            o["delta"] = (self.target - self.angles) % 4
        if self.geometry:
            cells, rots = forward_kinematics(self.angles)
            cells = cells - cells.mean(0)
            o["geometry"] = np.concatenate([cells, rots.reshape(-1, 9)], 1).astype(np.float32)
        return o

    def _draw_target(self):
        if self.targets is None:
            d = self.target_depth
            depth = int(self.np_random.integers(d[0], d[1] + 1)) if isinstance(d, tuple) else int(d)
            return near_shape(self.np_random, depth, self.num_wedges, self.executor)
        if callable(self.targets):
            return np.asarray(self.targets(self.np_random), dtype=np.int64)
        return self.targets[int(self.np_random.integers(len(self.targets)))].copy()

    # -- gym API --
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        for _ in range(100):
            self.target = np.asarray(options.get("target", self._draw_target()), dtype=np.int64)
            if "start" in options:
                self.angles = np.asarray(options["start"], dtype=np.int64).copy()
            elif self.curriculum_depth is not None:
                self.angles = random_walk(self.np_random, self.target, self.curriculum_depth, self.executor)
            else:
                self.angles = straight(self.num_wedges) if self.start is None else self.start.copy()
            if not self.is_success() or "target" in options:
                break
        self.max_steps = self.max_steps_default if self.curriculum_depth is None \
            else max(4 * self.curriculum_depth, 20)
        self.steps = 0
        self._mask = valid_move_mask(self.angles)
        self._blocked = np.zeros(2 * self.num_joints, dtype=bool)
        return self._obs(), {"distance": self.distance()}

    def step(self, action):
        action = int(action)
        joint, direction = action // 2, 1 - 2 * (action % 2)
        self.steps += 1
        phi = self._potential(self.angles)
        info = {"invalid": False, "blocked": False, "sim_time": 0.0}
        reward = STEP_PENALTY
        if not self._mask[action]:
            info["invalid"] = True
            reward += INVALID_PENALTY
        else:
            result = None if self.executor is None else self.executor.execute(self.angles, joint, direction)
            if result is None or result.done:
                self.angles = twist(self.angles, joint, direction)
                self._mask = valid_move_mask(self.angles)
                self._blocked[:] = False
            else:
                self._blocked[action] = True
                info["blocked"] = True
                reward += BLOCKED_PENALTY
            if result is not None:
                info["sim_time"] = result.sim_time
        success = self.is_success()
        reward += self.gamma * self._potential(self.angles) - phi
        if success:
            reward += SUCCESS_REWARD
        info.update(distance=self.distance(), success=success)
        truncated = not success and self.steps >= self.max_steps
        return self._obs(), float(reward), success, truncated, info
