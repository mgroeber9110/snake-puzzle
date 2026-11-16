import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rubiksnake.env import RubikSnakeEnv
from rubiksnake.kinematics import straight, valid_move_mask, valid_moves
from rubiksnake.search import astar
from rubiksnake.shapes import NAMED, is_flat_in_layer, named, random_flat_shape
from rubiksnake.sweep import is_twist_free

VECTORS = json.loads((Path(__file__).parents[2] / "test" / "vectors.json").read_text())
SWEEP_CONFIGS = [v for v in VECTORS["configs"] if "freeMoves" in v]


def test_mask_matches_valid_moves():
    for v in VECTORS["configs"]:
        if "validMoves" in v:
            mask = valid_move_mask(v["angles"])
            moves = [(a // 2, 1 - 2 * (a % 2)) for a in np.flatnonzero(mask)]
            assert moves == valid_moves(v["angles"])


@pytest.mark.parametrize("v", SWEEP_CONFIGS, ids=lambda v: "".join(map(str, v["angles"])))
def test_sweep_port_matches_js(v):
    free = [[j, d] for j, d in v["validMoves"] if is_twist_free(v["angles"], j, d)]
    assert free == v["freeMoves"]


def test_mujoco_agrees_with_exact_sweep():
    from rubiksnake.executor import MujocoExecutor
    ex = MujocoExecutor()
    agree = total = 0
    for v in SWEEP_CONFIGS:
        if len(v["angles"]) != 23:
            continue
        free = {tuple(m) for m in v["freeMoves"]}
        for j, d in v["validMoves"]:
            agree += ex.execute(v["angles"], j, d).done == ((j, d) in free)
            total += 1
    assert agree / total > 0.95, f"{agree}/{total}"


def test_every_stick_move_completes_in_mujoco():
    from rubiksnake.executor import MujocoExecutor
    ex = MujocoExecutor()
    for j in range(23):
        for d in (1, -1):
            r = ex.execute(straight(), j, d)
            assert r.done and 0.3 < r.sim_time < 0.8


@pytest.mark.parametrize("executor", ["none", "kinematic", "mujoco"])
def test_env_checker(executor):
    check_env(RubikSnakeEnv(executor=executor, target_depth=2), skip_render_check=True)


def test_reversed_reading_counts_as_success():
    target = named("wolf")
    env = RubikSnakeEnv(executor="none", targets=[target])
    env.reset(seed=0, options={"start": target[::-1]})
    assert env.is_success() and env.distance() == 0


def test_astar_solution_reaches_target():
    env = RubikSnakeEnv(executor="kinematic", target_depth=5)
    env.reset(seed=3)
    moves = astar(env.angles, env.target, env.executor)
    assert moves is not None and len(moves) <= 5
    for joint, direction in moves:
        _, _, term, _, info = env.step(2 * joint + (0 if direction == 1 else 1))
        assert not info["blocked"]
    assert term


def test_named_shapes_and_flat_layers():
    assert is_flat_in_layer(straight())
    assert is_flat_in_layer(named("wolf"))
    assert not is_flat_in_layer(named("ball"))
    assert not is_flat_in_layer(named("snowflake"))  # flat in a diagonal plane only
    assert is_flat_in_layer(random_flat_shape(np.random.default_rng(0)))
    assert set(NAMED) == {"wolf", "cat", "snowflake", "cobra", "ball"}
