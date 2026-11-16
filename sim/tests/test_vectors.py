"""Checks the Python port against test/vectors.json from the JS reference."""

import json
from pathlib import Path

import numpy as np
import pytest

from rubiksnake.kinematics import find_collisions, forward_kinematics, is_valid, valid_moves
from rubiksnake.notation import parse_notation, to_notation

VECTORS = json.loads((Path(__file__).parents[2] / "test" / "vectors.json").read_text())
CONFIGS = VECTORS["configs"]


@pytest.mark.parametrize("v", CONFIGS, ids=lambda v: "".join(map(str, v["angles"])))
def test_config(v):
    cells, rots = forward_kinematics(v["angles"])
    assert cells.tolist() == v["positions"]
    assert rots.reshape(len(rots), 9).tolist() == v["rotations"]
    assert [list(c) for c in find_collisions(v["angles"])] == v["collisions"]
    assert is_valid(v["angles"]) == v["valid"]
    if "validMoves" in v:
        assert [list(m) for m in valid_moves(v["angles"])] == v["validMoves"]
    if "notation" in v:
        assert to_notation(v["angles"]) == v["notation"]
        assert parse_notation(v["notation"], len(v["angles"]) + 1).tolist() == v["angles"]


@pytest.mark.parametrize("name", sorted(VECTORS["shapes"]))
def test_shapes(name):
    s = VECTORS["shapes"][name]
    angles = parse_notation(s["notation"])
    assert angles.tolist() == s["angles"]
    assert to_notation(angles) == s["canonical"]
    assert is_valid(angles) == s["valid"]
