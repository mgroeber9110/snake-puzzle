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

- MuJoCo against the exact sweep (`sweep.js`) on 1046 end-state-valid moves (test vectors, the Ball path, random
  walks of 10-60 moves, flat shapes): 1043 agree (99.7 %); 3 moves complete in MuJoCo although the exact check finds
  a collision, none is blocked although free. About 6 ms per twist on one core; a free twist takes 0.47-0.68 s of
  simulated time.
- This needed stiff servos: every joint that is not being twisted is held by a stiff spring (`hold_stiffness`), a
  twist that still pushes another joint by more than 0.05 rad counts as blocked, and the torque limit went up to 40 so
  the motor can turn 23 rigid wedges in time. With the first, soft servos (the stage-1 run) only 94.3 % of this set
  agreed, and 49 moves squeezed through by bending the rest of the snake.
- The exact sweep port takes about 5 ms per twist.

## First training run (stage 1, near shapes)

`runs/stage1/` holds the model, the curriculum log, the evaluation (`eval.md`) and example trajectories in the
exchange format of the draft. Run with `executor="mujoco"`, `ChainPolicy`, 4 environments on 4 cores.

- **Curriculum:** targets 1..D completed random moves from the stick, D grows at 85 % success. D went from 1 to 30
  (the cap) in about 21 minutes and 150k steps. Most of the time is spent learning depth 1; once the network has
  learned to compare angles and targets per joint, every further step takes about a minute.
- **Near shapes** (50 targets per depth, deterministic policy): 96–100 % success up to depth 30, random agent 0 %
  beyond depth 1. Where A* finds the optimum within its budget, the policy needs the same number of moves
  (ratio 1.0–1.06). A* solves only 4 % of the depth-30 targets within 5000 expansions, the policy all of them.
- **Stages 2 and 3 without further training**, from the stick, 200 steps:

  | Target | MuJoCo executor | Exact sweep check |
  |---|---|---|
  | Cobra | reached in 18 moves (optimal) | reached in 18 moves |
  | Snowflake | reached in 23 moves (optimal) | reached in 24 moves, 1 blocked |
  | Ball | reached in 23 moves (optimal) | stuck 3 moves away |
  | Wolf | stuck 5 moves away | stuck 7 moves away |
  | Cat | stuck 7 moves away | stuck 6 moves away |
  | Generated flat shapes (6 half turns) | 28 % | – |

**Caveats (fixed in stage 2 below).** The Ball only works in MuJoCo: 4 of its 23 moves pass in MuJoCo although the exact check finds an
overlap right at the start of the twist. MuJoCo lets them through by bending other joints by up to 0.78 rad, so the
made-up servos are too soft for a snake without play. Tightening this with a drift limit alone did not help (it also
blocks free moves, which bend joints through inertia); stiffer servos with a lower torque limit are the next thing to
try. Half-turn shapes (Wolf, Cat, generated flat ones) are the real gap: random walks from the stick rarely need a
joint at a half turn whose quarter-turn intermediate collides, so stage 2 needs its own training.

## Stage 2 (flat shapes)

**What can be reached at all.** With sharp-edged wedges and no play, two neighbouring half turns between straight
joints (`0,2,2,0`) cannot be reached: the last quarter turn is always blocked on the way, because a joint at a half
turn blocks its neighbours (corners overlap by about 0.15 cells). A breadth-first search over all states of a 6-wedge
and an 8-wedge snake with the exact check confirms it (912 of 920 and 12304 of 12585 valid states are reachable), and
what is unreachable for a piece of the snake is unreachable for the whole snake. The Wolf has this pattern three times
and the Cat as well, so neither can be reached in this model. With a clearance of 0.15 cells in the exact check all
920 states of the 6-wedge snake become reachable. Snowflake, Cobra and Ball contain no such pattern.

**Targets.** `target_kind="flat"` builds flat shapes by random half turns from the stick, each carried out by the
executor, so they are reachable by construction and lie in one lattice layer. `"mixed"` adds 25 % near shapes.

**Run.** `runs/stage2/`: the stage-1 model trained on with the stiff servos and mixed targets
(`--target-kind mixed --start-depth 4 --max-depth 14 --init runs/stage1/model_final.zip --lr 1e-4`), stopped after
67k steps (12 minutes) once 14 half turns had stayed at 99-100 %. Evaluation: `python train/evaluate_stage2.py
runs/stage2/model_final.zip`, results in `runs/stage2/eval.md`.

- Generated flat shapes up to 14 half turns: 100 %, in the minimum number of moves.
- Near shapes up to depth 30: 92-100 % with the most likely action, 100 % when actions are sampled. The failures of
  the deterministic policy are loops between a few states; sampling breaks them.
- Snowflake: reached in 26 moves (3 blocked tries), Cobra in 18, in MuJoCo and under the exact check alike.
- Wolf and Cat: the policy gets within 3 and 5 moves, which is as close as the model allows for the Wolf (one joint
  of each of its three neighbouring pairs stays off).
- Ball: stuck 2 moves away (1 when sampling). It is the open problem for stage 3: only one move leads out of the
  finished Ball (joint 20), so the last moves have to come in one particular order. A* from the Ball back to the stick
  did not finish within 60000 expansions, so whether the Ball is reachable with sharp edges is still open.
