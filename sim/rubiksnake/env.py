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
from .shapes import compact_walk, flat_walk, near_shape, random_walk

STEP_PENALTY = -0.01
SUCCESS_REWARD = 1.0
BLOCKED_PENALTY = -0.05
INVALID_PENALTY = -1.0
REVISIT_PENALTY = -0.05


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

    targets: None for generated targets: `target_kind="near"` (stage 1,
        `target_depth` random moves from the stick), `"flat"` (stage 2,
        `target_depth` random half turns from the stick, all in one lattice
        layer) or `"mixed"`; or a sequence of angle arrays / Wolter strings, or a callable
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
                 geometry: bool = False, delta: bool = True,
                 target_kind: str = "near", near_share: float = 0.25,
                 named_targets: dict | None = None, start_bank: dict | None = None,
                 named_share: float = 0.5, revisit: bool = False,
                 target_mix: dict | None = None):
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
        if target_kind not in ("near", "flat", "mixed", "mix"):
            raise ValueError(f"unknown target_kind {target_kind!r}")
        self.target_kind = target_kind
        self.near_share = near_share
        # target_kind="mix": draw the kind of each generated target from these weights
        self.target_mix = target_mix or {"near": 0.35, "flat": 0.3, "compact": 0.35}
        self.revisit = revisit
        # Stage 3: named shapes mixed into the generated targets. `start_bank`
        # maps a name to paths (lists of states) leading from the shape towards
        # the stick; with `curriculum_depth=N` an episode for that shape starts
        # up to N states along such a path instead of at the stick.
        self.named_targets = {k: np.asarray(v, dtype=np.int64) for k, v in (named_targets or {}).items()}
        self.start_bank = {k: [np.asarray(path, dtype=np.int64) for path in v]
                           for k, v in (start_bank or {}).items()}
        self.named_share = named_share
        self.episode_name = None
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
        if revisit:
            # Moves that would lead back to a shape already visited in this
            # episode. Lets the network learn to avoid going round in circles.
            obs["revisit"] = spaces.MultiBinary(2 * n)
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
        if self.revisit:
            o["revisit"] = self._revisit_flags().astype(np.int8)
        if self.delta:
            o["delta"] = (self.target - self.angles) % 4
        if self.geometry:
            cells, rots = forward_kinematics(self.angles)
            cells = cells - cells.mean(0)
            o["geometry"] = np.concatenate([cells, rots.reshape(-1, 9)], 1).astype(np.float32)
        return o

    def _revisit_flags(self) -> np.ndarray:
        flags = np.zeros(2 * self.num_joints, dtype=bool)
        for a in np.flatnonzero(self._mask):
            nxt = twist(self.angles, int(a // 2), 1 - 2 * int(a % 2))
            flags[a] = nxt.tobytes() in self._visited
        return flags

    def _draw_target(self):
        if self.targets is None:
            d = self.target_depth
            depth = int(self.np_random.integers(d[0], d[1] + 1)) if isinstance(d, tuple) else int(d)
            kind = self.target_kind
            if kind == "mix":
                kinds = sorted(self.target_mix)
                w = np.array([self.target_mix[k] for k in kinds], dtype=float)
                kind = kinds[int(self.np_random.choice(len(kinds), p=w / w.sum()))]
                if kind == "named":  # one of named_targets, from the start position
                    names = sorted(self.named_targets)
                    return self.named_targets[names[int(self.np_random.integers(len(names)))]].copy()
                if kind == "compact":
                    return compact_walk(self.np_random, self.executor, steps=60, num_wedges=self.num_wedges)
                if kind == "flat":
                    return flat_walk(self.np_random, depth, self.num_wedges, self.executor)
                return near_shape(self.np_random, min(2 * depth, 30), self.num_wedges, self.executor)
            if kind == "mixed":  # mostly flat shapes, some near shapes so stage 1 is not forgotten
                kind = "near" if self.np_random.random() < self.near_share else "flat"
            if kind == "flat":   # stage 2: `depth` half turns from the stick
                return flat_walk(self.np_random, depth, self.num_wedges, self.executor)
            if self.target_kind == "mixed":
                depth = min(2 * depth, 30)
            return near_shape(self.np_random, depth, self.num_wedges, self.executor)
        if callable(self.targets):
            return np.asarray(self.targets(self.np_random), dtype=np.int64)
        return self.targets[int(self.np_random.integers(len(self.targets)))].copy()

    # -- gym API --
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.episode_name = None
        if self.named_targets and "target" not in options and self.np_random.random() < self.named_share:
            return self._reset_named()
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
        self.max_steps = self.max_steps_default if self.curriculum_depth is None or self.named_targets \
            else max(4 * self.curriculum_depth, 20)
        return self._begin()

    def _reset_named(self):
        names = sorted(self.named_targets)
        name = names[int(self.np_random.integers(len(names)))]
        self.episode_name = name
        self.target = self.named_targets[name].copy()
        paths = self.start_bank.get(name)
        if paths and self.curriculum_depth is not None:
            path = paths[int(self.np_random.integers(len(paths)))]
            last = min(self.curriculum_depth, len(path) - 1)
            # half of the episodes at the frontier, the rest anywhere closer to the target
            k = last if self.np_random.random() < 0.5 else int(self.np_random.integers(1, last + 1))
            self.angles = path[k].copy()
            self.max_steps = 3 * k + 20
        else:
            self.angles = straight(self.num_wedges)
            self.max_steps = self.max_steps_default
        return self._begin()

    def _begin(self):
        self.steps = 0
        self._mask = valid_move_mask(self.angles)
        self._blocked = np.zeros(2 * self.num_joints, dtype=bool)
        self._visited = {self.angles.tobytes()}
        return self._obs(), {"distance": self.distance()}

    def step(self, action):
        action = int(action)
        joint, direction = action // 2, 1 - 2 * (action % 2)
        self.steps += 1
        phi = self._potential(self.angles)
        info = {"invalid": False, "blocked": False, "revisit": False, "sim_time": 0.0}
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
                key = self.angles.tobytes()
                info["revisit"] = key in self._visited
                if info["revisit"] and self.revisit:
                    reward += REVISIT_PENALTY
                self._visited.add(key)
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
        info.update(distance=self.distance(), success=success, named=self.episode_name)
        truncated = not success and self.steps >= self.max_steps
        return self._obs(), float(reward), success, truncated, info
