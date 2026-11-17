"""Build test sets of shapes the trained networks have never seen.

    python train/unseen_shapes.py --out runs/unseen/shapes.json

Every shape is made by twists from the straight stick that the MuJoCo executor
completes, so each one is reachable by construction. Categories:

- compact3d: walks biased towards a small radius of gyration (ball-like
  shapes, like the stage-3 targets but different ones)
- far: unbiased random walks of 60 and 100 moves (training went up to 30)
- flat: flat shapes with 18 and 24 half turns (training went up to 14)

Shapes closer than 8 quarter turns to one of the trained named shapes
(Ball, Cobra, Snowflake, in either reading) are dropped.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rubiksnake.kinematics import angle_distance, forward_kinematics, straight, twist, valid_move_mask
from rubiksnake.notation import to_notation
from rubiksnake.shapes import flat_walk, named, random_walk

TRAINED = ["ball", "cobra", "snowflake"]


def gyration(angles) -> float:
    cells, _ = forward_kinematics(angles)
    c = cells - cells.mean(0)
    return float(np.sqrt((c * c).sum(1).mean()))


def occupied_cells(angles) -> int:
    cells, _ = forward_kinematics(angles)
    return len({tuple(c) for c in cells.tolist()})


def compact_walk(rng, executor, steps=80, beta=3.0):
    """Random walk that prefers moves which make the shape more compact; returns
    the most compact state seen after at least 20 moves."""
    a = straight()
    best, best_rg = None, np.inf
    for k in range(steps):
        acts = np.flatnonzero(valid_move_mask(a))
        cand = [(int(x // 2), 1 - 2 * int(x % 2)) for x in acts]
        rg = np.array([gyration(twist(a, j, d)) for j, d in cand])
        p = np.exp(-beta * (rg - rg.min()))
        order = rng.choice(len(cand), size=len(cand), replace=False, p=p / p.sum())
        for i in order:
            j, d = cand[i]
            if executor.execute(a, j, d).done:
                a = twist(a, j, d)
                break
        else:
            break
        if k >= 20 and gyration(a) < best_rg:
            best, best_rg = a.copy(), gyration(a)
    return best


def far_from_trained(a, min_dist=8):
    for n in TRAINED:
        t = named(n)
        if min(angle_distance(a, t), angle_distance(a, t[::-1])) < min_dist:
            return False
    return True


def make(args):
    kind, seed = args
    from rubiksnake.executor import MujocoExecutor
    ex = MujocoExecutor()
    rng = np.random.default_rng(seed)
    if kind == "compact3d":
        a = compact_walk(rng, ex)
    elif kind.startswith("far"):
        a = random_walk(rng, straight(), int(kind[3:]), ex)
    else:
        a = flat_walk(rng, int(kind[4:]), executor=ex)
    return kind, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=20)
    ap.add_argument("--out", default="runs/unseen/shapes.json")
    args = ap.parse_args()
    kinds = ["compact3d", "far60", "far100", "flat18", "flat24"]
    # three times as many compact candidates, the most compact ones are kept
    jobs = [(k, 1000 * i + s) for i, k in enumerate(kinds)
            for s in range(args.per_kind * (3 if k == "compact3d" else 1))]
    with Pool(4) as pool:
        results = pool.map(make, jobs, chunksize=1)
    shapes, seen = [], set()
    counts = {}
    # most compact first among compact3d candidates
    results.sort(key=lambda r: (r[0], gyration(r[1]) if r[1] is not None else 99))
    for kind, a in results:
        if a is None or counts.get(kind, 0) >= args.per_kind:
            continue
        key = min(tuple(a.tolist()), tuple(a[::-1].tolist()))
        if key in seen or not far_from_trained(a) or angle_distance(a, straight()) < 8:
            continue
        seen.add(key)
        counts[kind] = counts.get(kind, 0) + 1
        shapes.append({"kind": kind, "notation": to_notation(a), "angles": a.tolist(),
                       "distance_from_stick": angle_distance(a, straight()),
                       "gyration": round(gyration(a), 2), "cells": occupied_cells(a)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(shapes, indent=1))
    ref = {n: (round(gyration(named(n)), 2), occupied_cells(named(n))) for n in TRAINED}
    print("reference (gyration, cells):", ref)
    for k in sorted(counts):
        sel = [s for s in shapes if s["kind"] == k]
        print(k, len(sel), "gyration", np.round(np.mean([s["gyration"] for s in sel]), 2),
              "cells", np.mean([s["cells"] for s in sel]), "dist", np.mean([s["distance_from_stick"] for s in sel]))


if __name__ == "__main__":
    main()
