"""Collect paths from named shapes back to the straight stick (for the stage-3 reverse curriculum).

    python train/make_unfold_bank.py runs/stage2/model_final.zip --shapes ball,cobra,snowflake --out runs/stage3/bank.json

The goal-conditioned policy is asked to *unfold* each shape: start at the
shape, target the stick, actions sampled. Every twist is reversible, so a
successful unfolding read backwards is a way to build the shape. The bank
stores the states along each path, starting at the shape.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def collect(seed, model_path, name, episodes, executor):
    import torch
    from sb3_contrib import MaskablePPO

    from rubiksnake.env import RubikSnakeEnv
    from rubiksnake.kinematics import straight
    from rubiksnake.shapes import named
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = MaskablePPO.load(model_path)
    env = RubikSnakeEnv(executor=executor, max_steps=150)
    paths = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 * seed + ep, options={"start": named(name), "target": straight()})
        states = [env.angles.tolist()]
        while True:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, _, term, trunc, _ = env.step(int(a))
            if env.angles.tolist() != states[-1]:
                states.append(env.angles.tolist())
            if term or trunc:
                break
        if term:
            # drop loops: cut back whenever a state repeats
            seen, clean = {}, []
            for st in states:
                key = tuple(st)
                if key in seen:
                    del clean[seen[key] + 1:]
                    seen = {tuple(c): i for i, c in enumerate(clean)}
                else:
                    seen[key] = len(clean)
                    clean.append(st)
            paths.append(clean)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--shapes", default="ball,cobra,snowflake")
    ap.add_argument("--episodes", type=int, default=40, help="per worker and shape")
    ap.add_argument("--executor", default="mujoco")
    ap.add_argument("--out", default="runs/stage3/bank.json")
    args = ap.parse_args()
    bank = {}
    with Pool(4) as pool:
        for name in args.shapes.split(","):
            res = pool.map(partial(collect, model_path=args.model, name=name, episodes=args.episodes,
                                   executor=args.executor), range(4))
            paths = {json.dumps(p) for r in res for p in r}
            bank[name] = [json.loads(p) for p in sorted(paths)]
            lens = [len(p) - 1 for p in bank[name]]
            print(f"{name}: {len(lens)} distinct paths of {4 * args.episodes} tries, "
                  f"moves min {min(lens, default=None)} mean {np.mean(lens) if lens else None}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(bank))


if __name__ == "__main__":
    main()
