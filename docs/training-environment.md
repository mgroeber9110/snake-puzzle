# Draft: Training environment for the Rubik's Snake

Status: draft for discussion · updated 2026-11-03 to the kinematic model on `main` (`src/kinematics.js`, `src/sweep.js`)

The goal is a Gym-style environment (Gymnasium API) in which a neural network learns to control the Rubik's Snake by twisting its joints. Training runs headless against the pure kinematic model; the Three.js page (`src/snake3.html`) stays the visualization.

## 1. Starting point in the repo

Since iterations 1 and 2, `main` has gained a rendering-free kinematic model that already covers most of what the environment needs:

- **`src/kinematics.js`**: 24 wedges, 23 joints (`0..22`, joint `j` connects wedge `j` and `j+1`), angles in quarter turns `0..3`, all zeros is the straight stick. `forwardKinematics(angles)` returns an exact integer cell and rotation per wedge. `findCollisions` / `isValid` are exact: two wedges may share a cell only as complementary halves, so shapes like the ball are accepted. `validMoves(angles)` lists all quarter-turn twists with a valid end state. `twist(angles, joint, direction)` applies a move.
- **`src/sweep.js`**: `findSweepCollision(angles, joint, direction)` samples the rotation (no vertex moves more than 0.05 between samples) and tests the real prism geometry with the separating axis theorem. `isTwistFree` combines the end state and the sweep check.
- **`test/`**: `npm test` (Node 22, no dependencies) checks the model against a brute-force reference, including a case where a twist is blocked on the way although its end state is valid.
- **`src/snake3.html`**: uses both modules for placement and collision checks and has the random auto-play from iteration 2.

The two weaknesses of `snake2.html` noted in the first draft (bounding-box collisions that reject legal shared cells, no check during the twist) are fixed there.

### Measured cost

A quick benchmark on a valid configuration 30 random moves away from the stick (Node 22, one core, cloud container):

| Check | Time per state | Result in that state |
|---|---|---|
| End state only (`validMoves`) | about 0.3 ms | 37 of 46 moves valid |
| End state and sweep (`isTwistFree` for each valid move) | about 70 ms | 20 of 46 moves free |

Two consequences shape the design below:

1. **The sweep matters.** In this state almost half of the moves that look valid are blocked on the way. A policy trained only on end-state validity would learn many moves a real snake cannot make.
2. **The sweep is too slow to run naively.** At about 70 ms per action mask, one core manages roughly 14 environment steps per second. RL needs several orders of magnitude more, so section 3 proposes how to make it cheaper.

## 2. Decision: Python for training, JavaScript stays the reference

**Recommendation: write the environment in Python and port the kinematic model to it.** `kinematics.js` remains the reference implementation and drives the browser.

| | Python | JavaScript (Node) |
|---|---|---|
| RL libraries | Gymnasium, Stable-Baselines3, sb3-contrib (action masking), PyTorch, RLlib | TF.js, hardly any ready-made RL algorithms |
| Parallelism | `VectorEnv`, multiprocessing, later vectorized NumPy/JAX | worker threads, more manual work |
| Physics for phase 4 | MuJoCo, PyBullet | hardly usable |
| Kinematic model | needs a port | exists and is tested |
| Visualization | weak | exists (`snake3.html`) |

The port is small (about 200 lines for the model and 150 for the sweep), and the integer arithmetic carries over one to one. Driving `kinematics.js` from Python through a Node subprocess was considered and rejected: the round trip per step costs more than the computation itself.

To keep both implementations identical, a small script exports **test vectors** from the JS model (`test/vectors.json`: angles → cells, rotations, collisions, valid moves, free moves). The Python tests check against them, and `npm test` regenerates and checks them, so every change to `kinematics.js` shows up on the Python side.

## 3. Kinematics in Python

### 3.1 API

The Python module follows the JS names, so code and tests can be compared directly:

```python
# sim/rubiksnake/kinematics.py   (port of src/kinematics.js)
NUM_WEDGES = 24
def forward_kinematics(angles) -> tuple[np.ndarray, np.ndarray]:  # (n,3) int cells, (n,3,3) int rotations
def find_collisions(angles) -> list[tuple[int, int]]
def is_valid(angles) -> bool
def twist(angles, joint, direction) -> np.ndarray
def valid_moves(angles) -> list[tuple[int, int]]        # end state only
def straight(num_wedges=24) -> np.ndarray

# sim/rubiksnake/sweep.py         (port of src/sweep.js)
def find_sweep_collision(angles, joint, direction) -> tuple[float, tuple[int, int]] | None
def is_twist_free(angles, joint, direction) -> bool
```

The number of wedges is a parameter. Small snakes (8 or 12 wedges) are valuable for tests, exhaustive search and the curriculum.

### 3.2 Making the sweep affordable

Three steps, in this order, until the environment reaches several thousand steps per second per core:

1. **Lattice pre-filter.** During a quarter turn about joint `j`, every moving wedge stays within a known ring around the joint axis. Fixed wedges whose cells lie outside every such ring can never be hit and need no geometric test. In most states this drops nearly all pairs, and the SAT test only runs on the few near the joint.
2. **Cache.** A move's sweep result depends only on the configuration and the move. An LRU cache keyed on `(angles, joint, direction)` pays off because episodes in a curriculum revisit nearby states constantly.
3. **Precomputed table (if still needed).** Whether a moving wedge sweeps through a fixed wedge depends only on their relative pose (cell offset and both rotations relative to the joint axis). That set is finite and small, so it can be enumerated once with `sweep.js` as the reference and stored as a lookup table. The sweep check then becomes a lattice lookup like the end-state check.

The environment offers both modes (`collision="end_state"`, fast, and `collision="sweep"`, realistic). Given the measurements above, `sweep` should be the default for training as soon as it is fast enough; `end_state` is useful for quick experiments and as an ablation.

## 4. The Gym environment

```python
class RubikSnakeEnv(gymnasium.Env):
    def __init__(self, num_wedges=24, task="reach_shape", targets=None,
                 collision="sweep", max_steps=200, curriculum_depth=None): ...
```

### 4.1 Actions

`Discrete(2 · 23 = 46)`: action `a` means joint `a // 2`, direction `+1` or `-1` quarter turn (`a % 2`), exactly the `(joint, direction)` pairs of `twist()` and `validMoves()`.

- A half turn takes two steps. That matches a servo moving in 90° steps and keeps the action space small.
- **Action masking** instead of penalties: `action_masks()` returns the free moves, and training uses `MaskablePPO` (sb3-contrib). Without a mask an invalid action leaves the state unchanged, costs `-1` and sets `info["invalid"] = True`.
- Joint numbers and directions are the same as in `kinematics.js` and `snake3.html`, so no mapping is needed.

### 4.2 Observation

A `Dict` space, so policy networks can choose what to use:

| Key | Space | Content |
|---|---|---|
| `angles` | `MultiDiscrete([4]*23)` | current joint angles (one-hot for the network, 92 values) |
| `target` | `MultiDiscrete([4]*23)` | target configuration (goal-conditioned learning) |
| `geometry` | `Box(24, 3+9)` | optional: cell relative to the centre of mass plus rotation matrix per wedge |
| `action_mask` | `MultiBinary(46)` | free moves |

The angles alone determine the state fully. `geometry` is redundant but may make spatial collisions easier to learn (a candidate for an ablation).

### 4.3 Tasks and rewards

**Task A, `reach_shape` (start here):** reach a target shape from a start configuration.

- `-0.01` per step, `+1` on reaching the target, `terminated=True`.
- Potential-based shaping, which does not change the optimal policy: `r += γ·Φ(s') − Φ(s)` with `Φ(s) = −d(s, target)/23`, where `d` is the sum of cyclic angle distances `min(|Δ|, 4−|Δ|)`. This distance ignores collisions and therefore underestimates the real path, which is fine for shaping.
- Optional shape-based target: "same occupancy up to rigid motion", computed from `forward_kinematics`. More expensive, but allows targets that are only given as a shape.
- Symmetry: the same shape can be read from either end (reversed joint order, possibly with mirrored angles; the exact mapping will be checked against the model). A target also counts as reached in the reversed reading.

**Task B, `explore` (auxiliary):** reward for new valid configurations, useful for pre-training and for stress-testing the sweep check.

**Task C, `locomotion` (phase 4):** move the centre of mass in a direction. This needs gravity, ground contact and friction, so a physics engine (section 6). The pure kinematic model cannot represent it meaningfully.

### 4.4 Episodes and curriculum

- `reset()` starts from the straight stick by default and draws a target from `targets`.
- **Reverse curriculum:** with `curriculum_depth=N` an episode starts N random free moves away from the target. N grows once the success rate exceeds about 80 %. This addresses the sparse reward among 4^23 ≈ 7·10¹³ angle combinations (before removing collisions).
- `truncated=True` after `max_steps` (default 200, with a curriculum `4·N`).
- `info`: `distance`, `invalid`, `blocked_by` (`end_state` or `sweep`), `success`.

### 4.5 Baselines

Before training a network we need reference results:

1. A random agent with the mask (`snake3.html`'s auto-play is the same idea).
2. A*/BFS with the angle distance as heuristic, giving optimal solutions for short distances. These serve as a quality measure (agent steps / optimal steps) and as data for imitation learning.

## 5. Connection to the browser simulation

A shared exchange format for trajectories:

```json
{
  "num_wedges": 24,
  "start": [0, 0, 0, "... 23 values ..."],
  "moves": [[5, 1], [12, -1]],
  "meta": {"policy": "ppo-v1", "task": "reach_shape", "target": "ball"}
}
```

`moves` are `[joint, direction]` pairs as taken by `twist()`.

Changes in `src/`:

- Replay mode in `snake3.html`: load a trajectory from a file or `?replay=<url>` and play it with the existing `animateMove`.
- A target overlay: show the target shape as a transparent ghost next to the snake.
- Optional later: export the policy as ONNX and run it live in the browser with onnxruntime-web, replacing the random auto-play.

## 6. Outlook: from puzzle to robot (phase 4)

For real motion control a physics layer is added:

- A generator builds a MuJoCo model (MJCF) from the wedge geometry: 23 revolute joints, masses, friction, ground.
- Actions become continuous (target angle or velocity per servo, `Box(-1, 1, (23,))`) or stay discrete 90° commands executed by a controller.
- Reward for `locomotion`: progress of the centre of mass minus energy and collision costs.
- Sim-to-real: domain randomization (mass, friction, servo backlash) once hardware data is available.

The discrete environment stays useful as a high-level planner that proposes joint sequences for the physics layer to execute.

## 7. Proposed structure and order

```
sim/
  pyproject.toml          # rubiksnake; dependencies: numpy, gymnasium
  rubiksnake/
    kinematics.py         # port of src/kinematics.js
    sweep.py              # port of src/sweep.js + pre-filter, cache, table
    env.py                # RubikSnakeEnv
    shapes.py             # library of target shapes
    search.py             # BFS/A* baseline
  tests/
    test_vectors.py       # checks against test/vectors.json
  train/
    ppo_masked.py         # sb3-contrib MaskablePPO
test/
  vectors.json            # exported from the JS model
src/
  kinematics.js, sweep.js # unchanged, reference
  snake3.html             # + replay mode, target overlay
```

| Step | Result | Acceptance |
|---|---|---|
| 1 | Test vector export from JS | `npm test` green, `vectors.json` committed |
| 2 | Python port of the kinematics and sweep | Python tests match all vectors |
| 3 | Faster sweep (pre-filter, cache, table) | same results as `sweep.js`, several thousand masks per second |
| 4 | `RubikSnakeEnv` + baselines | `gymnasium.utils.env_checker` green, random agent and A* run |
| 5 | Replay in the browser | a trajectory from Python looks identical in `snake3.html` |
| 6 | MaskablePPO with curriculum | success rate per curriculum depth, compared to A* |
| 7 | MuJoCo model, `locomotion` | later, depending on hardware |

## 8. Open questions

1. **What should the network learn first:** reaching shapes (task A, the proposed start) or locomotion (task C)?
2. **Hardware:** is there a plan for actuation yet (one servo per joint, 90° steps only)? That decides whether phase 4 stays discrete or becomes continuous.
3. **Target shapes:** which shapes should go into `shapes.py`? Known figures from Rubik's Snake notation can be imported now that the angle convention is fixed by `kinematics.js`.
