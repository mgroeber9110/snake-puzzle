"""Evaluate a trained policy per target depth against a random agent and A*.

    python train/evaluate.py runs/stage1/model_final.zip --depths 1,2,4,8,12,16,20,25,30 --episodes 50

Writes <model dir>/eval.json, eval.md and example trajectories in the exchange
format of section 5 (trajectories/*.json).
"""

from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from rubiksnake.env import RubikSnakeEnv
from rubiksnake.notation import to_notation
from rubiksnake.search import astar
from rubiksnake.shapes import NAMED, named, random_flat_shape


def max_steps_for(depth: int) -> int:
    return 4 * depth + 10


def run_episode(env, policy, rng, seed, target=None):
    """One episode; also tracks the closest the snake got to the target."""
    opts = {} if target is None else {"target": target}
    obs, _ = env.reset(seed=seed, options=opts)
    start, target = env.angles.copy(), env.target.copy()
    moves, blocked, closest = [], 0, env.distance()
    while True:
        if policy is None:
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
        else:
            a, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
            a = int(a)
        obs, _, term, trunc, info = env.step(a)
        moves.append([a // 2, 1 - 2 * (a % 2)])
        blocked += info["blocked"]
        closest = min(closest, info["distance"])
        if term or trunc:
            return dict(success=bool(term), steps=len(moves), blocked=blocked, closest=closest,
                        final=info["distance"],
                        start=start, target=target, moves=moves)


def eval_depth(depth, args):
    torch.set_num_threads(1)
    model = MaskablePPO.load(args.model)
    out = Path(args.model).parent
    rng = np.random.default_rng(depth)
    env = RubikSnakeEnv(executor=args.executor, target_depth=depth, max_steps=max_steps_for(depth))
    ppo, rnd, ratio, opt_lens = [], [], [], []
    for ep in range(args.episodes):
        seed = 10_000 * depth + ep
        r = run_episode(env, model, rng, seed)
        ppo.append(r)
        rnd.append(run_episode(env, None, rng, seed))
        opt = astar(r["start"], r["target"], env.executor, max_expansions=args.astar_budget)
        if opt is not None:
            opt_lens.append(len(opt))
            if r["success"]:
                ratio.append(r["steps"] / max(1, len(opt)))
        if ep < 2:
            traj = {"num_wedges": 24, "start": r["start"].tolist(), "moves": r["moves"],
                    "meta": {"policy": "ppo-stage1", "task": "reach_shape", "depth": depth,
                             "target": to_notation(r["target"]), "success": r["success"]}}
            (out / "trajectories" / f"depth{depth:02d}_{ep}.json").write_text(json.dumps(traj))
    row = {
        "depth": depth,
        "ppo_success": np.mean([r["success"] for r in ppo]),
        "random_success": np.mean([r["success"] for r in rnd]),
        "ppo_steps": np.mean([r["steps"] for r in ppo if r["success"]]) if any(r["success"] for r in ppo) else None,
        "astar_steps": np.mean(opt_lens) if opt_lens else None,
        "astar_solved": len(opt_lens) / args.episodes,
        "steps_over_optimal": np.mean(ratio) if ratio else None,
        "blocked_per_episode": np.mean([r["blocked"] for r in ppo]),
    }
    row = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--depths", default="1-12")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--executor", default="mujoco")
    ap.add_argument("--astar-budget", type=int, default=5_000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if "-" in args.depths:
        lo, hi = map(int, args.depths.split("-"))
        depths = list(range(lo, hi + 1))
    else:
        depths = [int(d) for d in args.depths.split(",")]
    model = MaskablePPO.load(args.model)
    out = Path(args.model).parent
    (out / "trajectories").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    with Pool(args.workers) as pool:
        rows = sorted(pool.imap_unordered(partial(eval_depth, args=args), sorted(depths, reverse=True)),
                      key=lambda r: r["depth"])
    for row in rows:
        print(json.dumps(row), flush=True)
    # Stages 2 and 3, zero-shot: generated flat shapes and the named shapes, from the stick.
    env = RubikSnakeEnv(executor=args.executor, max_steps=200)
    flat = [run_episode(env, model, rng, 50_000 + k, random_flat_shape(np.random.default_rng(k)))
            for k in range(args.episodes // 2)]
    shape_rows = [{"shape": "generated flat (one layer, 6 half turns)",
                   "success": round(float(np.mean([r["success"] for r in flat])), 3),
                   "start_distance": None, "closest": round(float(np.mean([r["closest"] for r in flat])), 2),
                   "steps": None, "blocked": round(float(np.mean([r["blocked"] for r in flat])), 1)}]
    for name in NAMED:
        r = run_episode(env, model, rng, 0, named(name))
        traj = {"num_wedges": 24, "start": r["start"].tolist(), "moves": r["moves"],
                "meta": {"policy": "ppo-stage1", "task": "reach_shape",
                         "target": to_notation(named(name)), "target_name": name,
                         "success": r["success"]}}
        (out / "trajectories" / f"{name}.json").write_text(json.dumps(traj))
        env.reset(options={"target": named(name)})
        shape_rows.append({"shape": name, "success": r["success"], "start_distance": env.distance(),
                           "closest": r["closest"], "steps": r["steps"], "blocked": r["blocked"]})
        print(json.dumps(shape_rows[-1]), flush=True)
    (out / "eval.json").write_text(json.dumps({"depths": rows, "shapes": shape_rows}, indent=2))
    lines = ["| Depth | PPO success | Random success | PPO steps | A* steps | PPO / A* | Blocked per episode |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['depth']} | {r['ppo_success']:.0%} | {r['random_success']:.0%} | {r['ppo_steps']} | "
                     f"{r['astar_steps']} | {r['steps_over_optimal']} | {r['blocked_per_episode']} |")
    lines += ["", "| Target from the stick | Success | Start distance | Closest distance | Steps | Blocked moves |",
              "|---|---|---|---|---|---|"]
    for r in shape_rows:
        lines.append(f"| {r['shape']} | {r['success']} | {r['start_distance']} | {r['closest']} | "
                     f"{r['steps']} | {r['blocked']} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
