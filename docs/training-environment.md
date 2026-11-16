# Draft: Training environment for the Rubik's Snake

Status: draft for discussion · updated 2026-11-15

Decided so far: the first task is reaching shapes; there is no hardware yet, so the physics is hypothetical and meant for exploration; twists are discrete moves executed in a physics engine (the hybrid approach in section 3).

The goal is a Gym-style environment (Gymnasium API) in which a neural network learns to control the Rubik's Snake by twisting its joints. Training runs headless: the policy picks discrete twists, MuJoCo executes them with simulated motors, and the pure kinematic model checks end states and targets. The Three.js page (`src/snake3.html`) stays the visualization.

## 1. Starting point in the repo

Since iterations 1 and 2, `main` has gained a rendering-free kinematic model that already covers most of what the environment needs:

- **`src/kinematics.js`**: 24 wedges, 23 joints (`0..22`, joint `j` connects wedge `j` and `j+1`), angles in quarter turns `0..3`, all zeros is the straight stick. `forwardKinematics(angles)` returns an exact integer cell and rotation per wedge. `findCollisions` / `isValid` are exact: two wedges may share a cell only as complementary halves, so shapes like the ball are accepted. `validMoves(angles)` lists all quarter-turn twists with a valid end state. `twist(angles, joint, direction)` applies a move.
- **`src/sweep.js`**: `findSweepCollision(angles, joint, direction)` samples the rotation (no vertex moves more than 0.05 between samples) and tests the real prism geometry with the separating axis theorem. `isTwistFree` combines the end state and the sweep check.
- **`test/`**: `npm test` (Node 22, no dependencies) checks the model against a brute-force reference, including a case where a twist is blocked on the way although its end state is valid.
- **`src/notation.js`**: `parseNotation` / `toNotation` convert between joint angles and Wolter notation, so known figures (Ball, Snowflake, Cobra, ...) can be used as targets directly.
- **`src/snake3.html`**: uses these modules for placement, collision checks and a shape field, and has the random auto-play from iteration 2.

The two weaknesses of `snake2.html` noted in the first draft (bounding-box collisions that reject legal shared cells, no check during the twist) are fixed there.

### Measured cost

A quick benchmark on a valid configuration 30 random moves away from the stick (Node 22, one core, cloud container):

| Check | Time per state | Result in that state |
|---|---|---|
| End state only (`validMoves`) | about 0.3 ms | 37 of 46 moves valid |
| End state and sweep (`isTwistFree` for each valid move) | about 70 ms | 20 of 46 moves free |

Two consequences shape the design below:

1. **The sweep matters.** In this state almost half of the moves that look valid are blocked on the way. A policy trained only on end-state validity would learn many moves a real snake cannot make.
2. **The sweep is too slow to run naively.** At about 70 ms per action mask, one core manages roughly 14 environment steps per second. RL needs several orders of magnitude more, so section 3 lets a physics engine take over this job.

## 2. Decision: Python for training, JavaScript stays the reference

**Recommendation: write the environment in Python and port the kinematic model to it.** `kinematics.js` remains the reference implementation and drives the browser.

| | Python | JavaScript (Node) |
|---|---|---|
| RL libraries | Gymnasium, Stable-Baselines3, sb3-contrib (action masking), PyTorch, RLlib | TF.js, hardly any ready-made RL algorithms |
| Parallelism | `VectorEnv`, multiprocessing, later vectorized NumPy/JAX | worker threads, more manual work |
| Physics engine | MuJoCo, PyBullet | hardly usable |
| Kinematic model | needs a port | exists and is tested |
| Visualization | weak | exists (`snake3.html`) |

The port is small (about 200 lines for `kinematics.js` and 100 for `notation.js`), and the integer arithmetic carries over one to one. `sweep.js` does not need a port: MuJoCo takes over its job. Driving `kinematics.js` from Python through a Node subprocess was considered and rejected: the round trip per step costs more than the computation itself.

To keep both implementations identical, a small script exports **test vectors** from the JS model (`test/vectors.json`: angles → cells, rotations, collisions, valid moves, free moves, notation). The Python tests check against them, and `npm test` regenerates and checks them, so every change to `kinematics.js` shows up on the Python side.

## 3. Kinematics in Python and twists in MuJoCo

### 3.1 Kinematic model (Python port)

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

# sim/rubiksnake/notation.py     (port of src/notation.js)
def parse_notation(text, num_wedges=24) -> np.ndarray
def to_notation(angles) -> str
```

The number of wedges is a parameter. Small snakes (8 or 12 wedges) are valuable for tests, exhaustive search and the curriculum.

### 3.2 Hybrid: discrete moves, executed by physics

The policy still chooses discrete moves `(joint, ±1)`. Instead of checking the sweep geometrically, the environment lets MuJoCo carry the move out:

1. **Model.** A generator builds an MJCF model from the wedge geometry in `kinematics.js`: 24 bodies with the exact prism as a convex mesh, 23 hinge joints on the face normals, made-up masses. Gravity is off for shape reaching, so the snake floats and only its own contacts matter.
2. **Motors.** Each joint gets a position actuator (PD controller) with made-up gain and a torque limit. Contacts between wedges directly connected by a joint are excluded; all other pairs collide. The contact margin is kept small, so wedges stay sharp-edged as decided for the kinematic model.
3. **Executing a move.** The target of the chosen joint moves by 90°; the simulation runs until the joint is within a tolerance of the target or a time limit runs out (a few hundred physics steps).
4. **Outcome.**
   - **Done:** the joint reached its target. All joints are snapped to the nearest quarter turn, the integer state is updated, and the end state is checked with `is_valid` as a safety net.
   - **Blocked:** the joint stalled because of contact. The joint is driven back to its old angle, the state stays unchanged, the move costs a small penalty and `info["blocked"] = True`.
5. **Mask.** The exact end-state check (about 0.3 ms) still masks out moves whose end state is invalid. Moves blocked only on the way are not masked; the agent learns them from blocked attempts.

Because the motors are made up, the physics parameters (gain, torque limit, masses, friction) are part of the configuration and can be explored. A stronger motor does not push through other wedges, since contacts act as hard constraints, but weak motors and friction can make borderline moves fail.

**Cost (estimated, not measured):** a MuJoCo step for 24 bodies takes on the order of tens of microseconds, so one twist of a few hundred steps takes a few milliseconds. That is about ten times faster than one `sweep.js` check and much faster than a full sweep mask. The first prototype step measures this.

**Check against `sweep.js`:** on random states, a move blocked in MuJoCo should be blocked in `sweep.js` and vice versa. Disagreements show where the soft contact model or the controller deviates from the exact geometry; `sweep.js` remains the reference for that comparison.

### 3.3 Fallback

If MuJoCo turns out too slow or too inexact, the exact path stays available: speed up `sweep.js` with a lattice pre-filter (only fixed wedges near the joint's swept ring are tested), a cache keyed on `(angles, joint, direction)`, and a precomputed table of blocking relative poses. The environment hides this behind the same option (`executor="mujoco"` or `executor="kinematic"`).

## 4. The Gym environment

```python
class RubikSnakeEnv(gymnasium.Env):
    def __init__(self, num_wedges=24, task="reach_shape", targets=None,
                 executor="mujoco", physics=None, max_steps=200, curriculum_depth=None): ...
```

### 4.1 Actions

`Discrete(2 · 23 = 46)`: action `a` means joint `a // 2`, direction `+1` or `-1` quarter turn (`a % 2`), exactly the `(joint, direction)` pairs of `twist()` and `validMoves()`.

- A half turn takes two steps. That matches a servo moving in 90° steps and keeps the action space small.
- **Action masking** for invalid end states: `action_masks()` returns the moves with a valid end state, and training uses `MaskablePPO` (sb3-contrib). Moves blocked on the way are found by executing them (section 3.2) and cost a small penalty (`-0.05`). Without a mask an invalid action leaves the state unchanged, costs `-1` and sets `info["invalid"] = True`.
- Joint numbers and directions are the same as in `kinematics.js` and `snake3.html`, so no mapping is needed.

### 4.2 Observation

A `Dict` space, so policy networks can choose what to use:

| Key | Space | Content |
|---|---|---|
| `angles` | `MultiDiscrete([4]*23)` | current joint angles (one-hot for the network, 92 values) |
| `target` | `MultiDiscrete([4]*23)` | target configuration (goal-conditioned learning) |
| `geometry` | `Box(24, 3+9)` | optional: cell relative to the centre of mass plus rotation matrix per wedge |
| `action_mask` | `MultiBinary(46)` | moves with a valid end state |

The angles alone determine the state fully. `geometry` is redundant but may make spatial collisions easier to learn (a candidate for an ablation).

### 4.3 Tasks and rewards

**Task A, `reach_shape` (start here):** reach a target shape from a start configuration.

- `-0.01` per step, `+1` on reaching the target, `terminated=True`.
- Potential-based shaping, which does not change the optimal policy: `r += γ·Φ(s') − Φ(s)` with `Φ(s) = −d(s, target)/23`, where `d` is the sum of cyclic angle distances `min(|Δ|, 4−|Δ|)`. This distance ignores collisions and therefore underestimates the real path, which is fine for shaping.
- Targets come from `notation.py`, so `targets` can be given as Wolter strings.
- Optional shape-based target: "same occupancy up to rigid motion", computed from `forward_kinematics`. More expensive, but allows targets that are only given as a shape.
- Symmetry: the same shape can be read from either end (reversed joint order, possibly with mirrored angles; the exact mapping will be checked against the model). A target also counts as reached in the reversed reading.

**Task B, `explore` (auxiliary):** reward for new valid configurations, useful for pre-training and for stress-testing the twist executor.

**Task C, `locomotion` (later):** move the centre of mass in a direction. This needs gravity, ground contact and friction, which the MuJoCo model from section 3.2 can switch on (section 6).

### 4.4 Episodes and curriculum

- `reset()` starts from the straight stick by default and draws a target from `targets`.
- **Reverse curriculum:** with `curriculum_depth=N` an episode starts N random moves away from the target. The start is generated with moves the executor actually completes, so every start can reach the target. N grows once the success rate exceeds about 80 %. This addresses the sparse reward among 4^23 ≈ 7·10¹³ angle combinations (before removing collisions).
- **Target stages:** besides the depth N, the targets themselves get harder in three stages. A stage is added once the previous one is solved reliably:
  1. *Near shapes:* generated configurations a few moves from the straight stick.
  2. *Flat shapes:* fewer twists should be blocked on the way than in compact 3D shapes (an assumption to check). There are two kinds, checked with `forwardKinematics`:
     - *Flat in a lattice layer:* all wedge cells in one layer, like the straight stick. These use only half turns. Targets are generated by random half turns from the stick, each carried out by the executor (`flat_walk`). The Wolf (1x7x6 cells, 10 joints turned) and the Cat are not targets: with sharp-edged wedges two neighbouring half turns between straight joints (`0,2,2,0`) cannot be reached, and both shapes contain that pattern (see `sim/README.md`; decided 2026-11-16 to keep sharp edges).
     - *Flat in a diagonal plane:* the Snowflake, a hexagonal ring in a thin slab perpendicular to a cube diagonal (5x5x5 cells in the lattice). It turns all 23 joints by quarter turns and ends this stage.
  3. *3D shapes:* Cobra, then the Ball (a 3x3x3 block, the densest shape and presumably the one with the most blocked moves).
- `truncated=True` after `max_steps` (default 200, with a curriculum `4·N`).
- `info`: `distance`, `invalid`, `blocked`, `success`, and the simulated time a move took.

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

## 6. Outlook: from puzzle to robot

There is no hardware yet, so everything here stays hypothetical. The MuJoCo model from section 3.2 is extended step by step:

- Gravity, ground and friction switched on for `locomotion`.
- Actions become continuous (target angle or velocity per servo, `Box(-1, 1, (23,))`) or stay discrete 90° commands executed by a controller.
- Reward for `locomotion`: progress of the centre of mass minus energy and collision costs.
- Sim-to-real: domain randomization (mass, friction, servo backlash) if hardware is built later.

The discrete environment stays useful as a high-level planner that proposes joint sequences for the physics layer to execute.

## 7. Proposed structure and order

```
sim/
  pyproject.toml          # rubiksnake; dependencies: numpy, gymnasium, mujoco
  rubiksnake/
    kinematics.py         # port of src/kinematics.js
    notation.py           # port of src/notation.js
    mjcf.py               # MuJoCo model generator
    executor.py           # executes twists in MuJoCo (section 3.2)
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
  kinematics.js, sweep.js, notation.js  # unchanged, reference
  snake3.html             # + replay mode, target overlay
```

| Step | Result | Acceptance |
|---|---|---|
| 1 | Test vector export from JS | `npm test` green, `vectors.json` committed |
| 2 | Python port of kinematics and notation | Python tests match all vectors |
| 3 | MJCF generator + twist executor | time per twist measured; blocked moves agree with `sweep.js` on random states |
| 4 | `RubikSnakeEnv` + baselines | `gymnasium.utils.env_checker` green, random agent and A* run |
| 5 | Replay in the browser | a trajectory from Python looks identical in `snake3.html` |
| 6 | MaskablePPO with curriculum | success rate per curriculum depth, compared to A* |
| 7 | Gravity, ground, `locomotion` | later, for exploration |

## 8. Decisions on the former open questions

1. **Physics parameters:** the made-up motor values are chosen so a twist takes about 0.5 s of simulated time and the motor easily turns the longer half of the snake. They are explored from there.
2. **Target shapes:** near shapes first, then flat shapes (generated ones, ending with the Snowflake; the Wolf turned out to be unreachable with sharp edges), then 3D shapes (Cobra, then Ball), as listed in section 4.4.
