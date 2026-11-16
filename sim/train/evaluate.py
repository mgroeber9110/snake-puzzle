"""Evaluate a trained policy per target depth against a random agent and A*.

    python train/evaluate.py runs/stage1/model_final.zip --depths 1-12 --episodes 100

Writes <model dir>/eval.json, eval.md and example trajectories in the exchange
format of section 5 (trajectories/*.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from rubiksnake.env import RubikSnakeEnv
from rubiksnake.notation import to_notation
from rubiksnake.search import astar


def max_steps_for(depth: int) -> int:
    return 4 * depth + 10


def run_episode(env, policy, rng, seed, target=None):
    opts = {} if target is None else {"target": target}
    obs, _ = env.reset(seed=seed, options=opts)
    start, target = env.angles.copy(), env.target.copy()
    moves, blocked = [], 0
    while True:
        if policy is None:
            a = int(rng.choice(np.flatnonzero(env.action_masks())))
        else:
            a, _ = policy.predict(obs, action_masks=env.action_masks(), deterministic=True)
            a = int(a)
        obs, _, term, trunc, info = env.step(a)
        moves.append([a // 2, 1 - 2 * (a % 2)])
        blocked += info["blocked"]
        if term or trunc:
            return dict(success=bool(term), steps=len(moves), blocked=blocked,
                        start=start, target=target, moves=moves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--depths", default="1-12")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--executor", default="mujoco")
    ap.add_argument("--astar-budget", type=int, default=20_000)
    args = ap.parse_args()
    lo, hi = map(int, args.depths.split("-"))
    model = MaskablePPO.load(args.model)
    out = Path(args.model).parent
    (out / "trajectories").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for depth in range(lo, hi + 1):
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
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "eval.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Depth | PPO success | Random success | PPO steps | A* steps | PPO / A* | Blocked per episode |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['depth']} | {r['ppo_success']:.0%} | {r['random_success']:.0%} | {r['ppo_steps']} | "
                     f"{r['astar_steps']} | {r['steps_over_optimal']} | {r['blocked_per_episode']} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
