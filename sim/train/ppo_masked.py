"""Train MaskablePPO on curriculum stage 1 (near shapes) of RubikSnakeEnv.

Each episode starts at the straight stick; the target is a configuration
reached by 1..D random moves that the MuJoCo executor completes. D starts at 1
and grows by one whenever the success rate of the last episodes at the current
D exceeds a threshold (section 4.4 of docs/training-environment.md).

    python train/ppo_masked.py --steps 2000000 --out runs/stage1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rubiksnake.env import RubikSnakeEnv


def max_steps_for(depth: int) -> int:
    return 4 * depth + 10


class Curriculum(BaseCallback):
    def __init__(self, log_path: Path, start_depth=1, max_depth=16, window=500, threshold=0.85):
        super().__init__()
        self.depth = start_depth
        self.max_depth = max_depth
        self.window = window
        self.threshold = threshold
        self.recent = collections.deque(maxlen=window)
        self.log_path = log_path
        self.t0 = time.time()
        self.rows = []

    def _apply(self):
        self.training_env.env_method("set_target_depth", (1, self.depth))
        self.training_env.env_method("set_max_steps", max_steps_for(self.depth))

    def _on_training_start(self):
        self._apply()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.recent.append(bool(info.get("success")))
        if len(self.recent) == self.window:
            rate = sum(self.recent) / self.window
            row = {"timesteps": self.num_timesteps, "minutes": round((time.time() - self.t0) / 60, 1),
                   "depth": self.depth, "success_rate": round(rate, 3)}
            self.rows.append(row)
            print(json.dumps(row), flush=True)
            self.recent.clear()
            if rate >= self.threshold and self.depth < self.max_depth:
                self.depth += 1
                self._apply()
                self.model.save(self.log_path.parent / f"model_depth{self.depth - 1}")
            with open(self.log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerows(self.rows)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--executor", default="mujoco")
    ap.add_argument("--max-depth", type=int, default=16)
    ap.add_argument("--out", default="runs/stage1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(RubikSnakeEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"executor": args.executor, "target_depth": (1, 1),
                                   "max_steps": max_steps_for(1)})
    model = MaskablePPO("MultiInputPolicy", env, n_steps=512, batch_size=256, n_epochs=4,
                        learning_rate=3e-4, gamma=0.99, ent_coef=0.01, seed=args.seed,
                        policy_kwargs={"net_arch": [256, 256]}, verbose=0)
    cb = Curriculum(out / "curriculum.csv", max_depth=args.max_depth)
    t = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(out / "model_final")
    json.dump({"steps": args.steps, "seconds": round(time.time() - t), "final_depth": cb.depth,
               "executor": args.executor}, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
