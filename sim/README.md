# Training environment (Python)

Implements the design in [`docs/training-environment.md`](../docs/training-environment.md): a Gymnasium
environment in which a policy twists the joints of the snake, MuJoCo executes each twist with simulated
servos, and the kinematic model checks end states and targets.

```bash
cd sim
pip install -e ".[train,test]"
pytest                      # checks the port against ../test/vectors.json and the environment
python train/ppo_masked.py --steps 800000 --out runs/stage1
python train/evaluate.py runs/stage1/model_final.zip --depths 1-30
```

## Modules

| Module | Content |
|---|---|
| `rubiksnake/kinematics.py` | Port of `src/kinematics.js` (same names, snake_case). `valid_move_mask` gives the 46-move end-state mask in about 0.2 ms. |
| `rubiksnake/notation.py` | Port of `src/notation.js` (Wolter notation). |
| `rubiksnake/sweep.py` | Vectorized port of `src/sweep.js`, the exact check of the path of a twist. |
| `rubiksnake/mjcf.py` | MuJoCo model: 24 wedge bodies with the exact prism as convex mesh, hinges on the shared faces, a position servo per joint. All physics values are made up and live in `Physics`. |
| `rubiksnake/executor.py` | `MujocoExecutor` runs a twist (target ramps 90° in 0.4 s; blocked if the joint has not arrived after 0.8 s or another joint was pushed off). `KinematicExecutor` uses the exact sweep instead. Both cache results. |
| `rubiksnake/env.py` | `RubikSnakeEnv`, task `reach_shape`. |
| `rubiksnake/shapes.py` | Named shapes (Wolf, Cat, Snowflake, Cobra, Ball), near shapes (random walks from the stick), generated flat shapes. |
| `rubiksnake/search.py` | A* baseline with the angle distance as heuristic and lazy executor checks. |
| `rubiksnake/policy.py` | `ChainPolicy`: per-joint features, dilated 1D convolutions along the chain, one shared action head for all joints. |

The JS model stays the reference: `node scripts/export-vectors.js` writes `test/vectors.json`, `npm test` fails if
it is out of date, and `pytest` checks the Python port against it (forward kinematics, collisions, valid moves,
moves free along the whole sweep, notation).

## Environment

- **Actions:** `Discrete(46)`, action `a` twists joint `a // 2` by `+1` (even `a`) or `-1` quarter turn.
  `env.action_masks()` gives the moves with a valid end state (for `MaskablePPO`).
- **Observation** (`Dict`): `angles`, `target`, `delta` = `(target - angles) mod 4`, `action_mask`, `blocked`
  (moves tried in this state and blocked on the way; cleared when the state changes), optional `geometry`.
- **Reward:** `-0.01` per step, `+1` on success, `-0.05` for a blocked move, `-1` for a masked move, plus
  potential-based shaping with `Φ = -distance / 23`. The target also counts as reached when read from the other
  end of the snake, which is the reversed angle list.
- **Executors:** `executor="mujoco"` (default), `"kinematic"` (exact sweep) or `"none"` (end state only).
- **Curriculum:** `target_depth=(lo, hi)` draws stage-1 targets that many completed random moves from the stick;
  `curriculum_depth=N` starts N moves away from the target instead (reverse curriculum); `targets=[...]` takes angle
  lists or Wolter strings.

## Measured

- MuJoCo against the exact sweep (`sweep.js`) on the 303 end-state-valid moves of the test vectors: 296 agree (97.7 %);
  4 moves complete in MuJoCo although the exact check finds a collision, 3 are blocked although free. About 6–7 ms per
  twist on one core; a free twist takes 0.45–0.6 s of simulated time.
- The exact sweep port takes about 5 ms per twist.
