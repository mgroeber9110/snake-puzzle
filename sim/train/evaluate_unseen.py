"""Run trained networks on the unseen shapes from unseen_shapes.py.

    python train/evaluate_unseen.py runs/stage3/model_final.zip runs/stage2/model_final.zip

Every shape is tried from the straight stick with the MuJoCo executor, 200
steps: once with the most likely action and 8 times with sampled actions
("best of 8" counts a shape as reached if any sample reaches it).
Writes runs/unseen/eval.md, eval.json and the trajectories of the first network (most likely action).
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SAMPLES = 8


def run_shape(shape, model_path):
    import torch
    from sb3_contrib import MaskablePPO

    from rubiksnake.env import RubikSnakeEnv
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = MaskablePPO.load(model_path)
    env = RubikSnakeEnv(executor="mujoco", max_steps=200,
                         revisit="revisit" in model.observation_space.spaces)
    target = np.array(shape["angles"])
    out = []
    for k in range(1 + SAMPLES):
        obs, _ = env.reset(seed=k, options={"target": target})
        d0, closest, blocked, moves = env.distance(), env.distance(), 0, []
        while True:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=(k == 0))
            obs, _, term, trunc, info = env.step(int(a))
            moves.append([int(a) // 2, 1 - 2 * (int(a) % 2)])
            closest = min(closest, info["distance"])
            blocked += info["blocked"]
            if term or trunc:
                break
        out.append({"success": bool(term), "steps": len(moves), "distance": d0, "closest": closest,
                    "blocked": int(blocked), "moves": moves})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--shapes", default="runs/unseen/shapes.json")
    args = ap.parse_args()
    shapes = json.loads(Path(args.shapes).read_text())
    out_dir = Path(args.shapes).parent
    (out_dir / "trajectories").mkdir(exist_ok=True)
    kinds = list(dict.fromkeys(s["kind"] for s in shapes))
    report = {}
    lines = ["| Shapes | Network | Count | Reached (most likely action) | Reached (best of 8 samples) | "
             "Moves / distance | Closest when not reached | Blocked per attempt |", "|---|---|---|---|---|---|---|---|"]
    for m_idx, model_path in enumerate(args.models):
        name = Path(model_path).parent.name
        with Pool(4) as pool:
            res = pool.map(partial(run_shape, model_path=model_path), shapes, chunksize=1)
        report[name] = []
        for shape, r in zip(shapes, res):
            report[name].append({"kind": shape["kind"], "notation": shape["notation"],
                                 "attempts": [{k: v for k, v in a.items() if k != "moves"} for a in r]})
            if m_idx == 0:
                i = len(report[name]) - 1
                det = r[0]
                (out_dir / "trajectories" / f"{shape['kind']}_{i:02d}.json").write_text(json.dumps({
                    "num_wedges": 24, "start": [0] * 23, "moves": det["moves"],
                    "meta": {"policy": name, "task": "reach_shape", "target": shape["notation"],
                             "target_name": f"unseen {shape['kind']} #{i}", "executor": "mujoco",
                             "success": det["success"]}}))
        for kind in kinds:
            rows = [(s, r) for s, r in zip(shapes, res) if s["kind"] == kind]
            det = [r[0] for _, r in rows]
            best = [any(a["success"] for a in r) for _, r in rows]
            ok = [a for a in det if a["success"]]
            fail = [a for a in det if not a["success"]]
            ratio = f"{np.mean([a['steps'] / a['distance'] for a in ok]):.2f}" if ok else "-"
            closest = (f"{np.mean([a['closest'] for a in fail]):.1f} of {np.mean([a['distance'] for a in fail]):.1f}"
                       if fail else "-")
            blocked = np.mean([a["blocked"] for _, r in rows for a in r])
            lines.append(f"| {kind} | {name} | {len(rows)} | {len(ok)}/{len(rows)} | {sum(best)}/{len(rows)} | "
                         f"{ratio} | {closest} | {blocked:.1f} |")
            print(lines[-1], flush=True)
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")
    (out_dir / "eval.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
