"""Evaluate a policy on stage-2 targets: generated flat shapes, near shapes and the named shapes.

    python train/evaluate_stage2.py runs/stage2/model_final.zip

Every target is tried twice from the stick: with the most likely action
(deterministic) and with actions sampled from the policy. Named shapes are
run with the MuJoCo executor and with the exact sweep check. Writes
eval.md / eval.json and trajectories next to the model.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np


def episode(env, model, deterministic, seed, target=None):
    opts = {} if target is None else {"target": target}
    obs, _ = env.reset(seed=seed, options=opts)
    start, d0 = env.angles.copy(), env.distance()
    moves, blocked, closest = [], 0, d0
    while True:
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=deterministic)
        a = int(a)
        obs, _, term, trunc, info = env.step(a)
        moves.append([a // 2, 1 - 2 * (a % 2)])
        blocked += info["blocked"]
        closest = min(closest, info["distance"])
        if term or trunc:
            return dict(success=bool(term), steps=len(moves), blocked=int(blocked), distance=d0,
                        closest=int(closest), start=start.tolist(), moves=moves)


def _load(path):
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return MaskablePPO.load(path)


def eval_generated(arg):
    path, kind, depth, n = arg
    from rubiksnake.env import RubikSnakeEnv
    model = _load(path)
    env = RubikSnakeEnv(target_kind=kind, target_depth=depth, max_steps=8 * depth + 20)
    det = [episode(env, model, True, 1000 * depth + s) for s in range(n)]
    smp = [episode(env, model, False, 1000 * depth + s) for s in range(n)]
    ok = [r for r in det if r["success"]]
    return {"targets": f"{kind}, depth {depth}", "episodes": n,
            "mean_distance": round(float(np.mean([r["distance"] for r in det])), 1),
            "success_deterministic": round(len(ok) / n, 3),
            "success_sampled": round(float(np.mean([r["success"] for r in smp])), 3),
            "steps_per_distance": round(float(np.mean([r["steps"] / max(r["distance"], 1) for r in ok])), 2) if ok else None,
            "blocked_per_episode": round(float(np.mean([r["blocked"] for r in det])), 2)}


def eval_named(arg):
    path, name, executor = arg
    from rubiksnake.env import RubikSnakeEnv
    from rubiksnake.notation import to_notation
    from rubiksnake.shapes import named
    model = _load(path)
    env = RubikSnakeEnv(executor=executor, max_steps=200)
    out = {"shape": name, "executor": executor}
    for label, det in (("deterministic", True), ("sampled", False)):
        r = episode(env, model, det, 0, named(name))
        out[label] = {k: r[k] for k in ("success", "steps", "blocked", "distance", "closest")}
        out[label + "_trajectory"] = {"num_wedges": 24, "start": r["start"], "moves": r["moves"],
                                      "meta": {"policy": Path(path).parent.name, "task": "reach_shape",
                                               "target": to_notation(named(name)), "target_name": name,
                                               "executor": executor, "success": r["success"]}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from rubiksnake.shapes import NAMED
    out = Path(args.model).parent
    (out / "trajectories").mkdir(exist_ok=True)
    gen = [(args.model, "flat", d, args.episodes) for d in (2, 4, 8, 12, 14)] + \
          [(args.model, "near", d, args.episodes) for d in (4, 10, 20, 30)]
    nam = [(args.model, n, ex) for n in NAMED for ex in ("mujoco", "kinematic")]
    with Pool(args.workers) as pool:
        rows = pool.map(eval_generated, gen, chunksize=1)
        shapes = pool.map(eval_named, nam, chunksize=1)
    lines = ["| Targets | Mean distance | Success (deterministic) | Success (sampled) | Steps / distance | Blocked per episode |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['targets']} | {r['mean_distance']} | {r['success_deterministic']:.0%} | "
                     f"{r['success_sampled']:.0%} | {r['steps_per_distance']} | {r['blocked_per_episode']} |")
    lines += ["", "| Shape from the stick | Executor | Deterministic | Sampled |", "|---|---|---|---|"]

    def cell(r):
        return (f"reached in {r['steps']} moves ({r['blocked']} blocked)" if r["success"]
                else f"not reached, closest {r['closest']} of {r['distance']} ({r['blocked']} blocked)")
    for s in shapes:
        for label in ("deterministic", "sampled"):
            traj = s.pop(label + "_trajectory")
            if s["executor"] == "mujoco":
                (out / "trajectories" / f"{s['shape']}_{label}.json").write_text(json.dumps(traj))
        ex = "MuJoCo" if s["executor"] == "mujoco" else "exact sweep"
        lines.append(f"| {s['shape']} | {ex} | {cell(s['deterministic'])} | {cell(s['sampled'])} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n")
    (out / "eval.json").write_text(json.dumps({"generated": rows, "shapes": shapes}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
