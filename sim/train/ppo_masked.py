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

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rubiksnake.env import RubikSnakeEnv
from rubiksnake.policy import ChainPolicy


def max_steps_for(depth: int, kind: str = "near") -> int:
    # near: depth quarter turns; flat/mixed: depth half turns (or 2*depth quarter turns), with room for detours
    return 4 * depth + 10 if kind == "near" else 8 * depth + 20


class Curriculum(BaseCallback):
    def __init__(self, log_path: Path, start_depth=1, max_depth=16, window=500, threshold=0.85, kind="near"):
        super().__init__()
        self.depth = start_depth
        self.kind = kind
        self.max_depth = max_depth
        self.window = window
        self.threshold = threshold
        self.recent = collections.deque(maxlen=window)
        self.log_path = log_path
        self.t0 = time.time()
        self.rows = []

    def _apply(self):
        if self.kind == "named":  # stage 3: depth = how far along an unfolding path episodes start
            self.training_env.env_method("set_curriculum_depth", self.depth)
            return
        self.training_env.env_method("set_target_depth", (1, self.depth))
        self.training_env.env_method("set_max_steps", max_steps_for(self.depth, self.kind))

    def _on_training_start(self):
        self._apply()

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and (self.kind != "named" or info.get("named")):
                self.recent.append(bool(info.get("success")))
        if len(self.recent) == self.window:
            rate = sum(self.recent) / self.window
            row = {"timesteps": self.num_timesteps, "minutes": round((time.time() - self.t0) / 60, 1),
                   "depth": self.depth, "success_rate": round(rate, 3)}
            self.rows.append(row)
            print(json.dumps(row), flush=True)
            self.recent.clear()
            self.model.save(self.log_path.parent / "model_latest")
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
    ap.add_argument("--target-kind", default="near", choices=["near", "flat", "mixed"],
                    help="near = stage 1, flat = stage 2 (depth counts half turns), mixed = mostly flat")
    ap.add_argument("--start-depth", type=int, default=1)
    ap.add_argument("--init", help="continue from this model instead of a fresh network")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--named", help="stage 3: comma-separated named shapes mixed into the targets; the "
                    "curriculum depth then counts moves along the paths in --bank, the generated "
                    "targets keep --start-depth")
    ap.add_argument("--bank", help="paths from make_unfold_bank.py")
    ap.add_argument("--named-share", type=float, default=0.6)
    ap.add_argument("--named-start", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="the env processes need the cores; more torch threads slow the rollouts down")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"executor": args.executor, "target_depth": (1, args.start_depth),
                  "target_kind": args.target_kind,
                  "max_steps": max_steps_for(args.start_depth, args.target_kind)}
    if args.named:
        from rubiksnake.shapes import named
        names = args.named.split(",")
        bank = json.load(open(args.bank))
        env_kwargs.update(named_targets={n: named(n).tolist() for n in names},
                          start_bank={n: bank[n] for n in names}, named_share=args.named_share,
                          curriculum_depth=args.named_start)
    env = make_vec_env(RubikSnakeEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv,
                       env_kwargs=env_kwargs)
    if args.init:
        model = MaskablePPO.load(args.init, env=env, learning_rate=args.lr, seed=args.seed)
    else:
        model = MaskablePPO(ChainPolicy, env, n_steps=512, batch_size=256, n_epochs=4,
                            learning_rate=args.lr, gamma=0.99, ent_coef=0.01, seed=args.seed, verbose=0)
    if args.named:
        cb = Curriculum(out / "curriculum.csv", start_depth=args.named_start, max_depth=args.max_depth,
                        kind="named", threshold=0.9)
    else:
        cb = Curriculum(out / "curriculum.csv", start_depth=args.start_depth, max_depth=args.max_depth,
                        kind=args.target_kind)
    t = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(out / "model_final")
    json.dump({"steps": args.steps, "seconds": round(time.time() - t), "final_depth": cb.depth,
               "executor": args.executor, "target_kind": args.target_kind, "init": args.init}, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
