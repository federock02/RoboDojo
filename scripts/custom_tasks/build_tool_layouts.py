"""Build the checked-in reference layouts for the two L-tool tasks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "task/RoboDojo/tasks"))

from l_tool_asset import (
    ARM_CENTER_OFFSET_M,
    ARM_HALF_WIDTH_M,
    CROSSBAR_INSET_M,
    HANDLE_CENTER_Y_M,
    TOOL_HEIGHT_M,
    TOOL_LENGTH_M,
    TOOL_MASS_KG,
    TOOL_WIDTH_M,
)
from seeded_task_variants import (
    REFERENCE_CUBE_ASSET_SIDE_M,
    TOOL_CONTACT_GAP_M,
    TOOL_PULL_TASK,
    TOOL_PUSH_TASK,
)

CUBE_SIDE_M = 0.06
CUBE_MASS_KG = 0.1
TARGET_SIDE_M = 0.16
NATIVE_TARGET_SIDE_M = 0.2233600115334982


def _set_limits(obj):
    for axis, value in zip(("xlim", "ylim", "zlim"), obj["default_pos"], strict=True):
        obj[axis] = [value, value]


def tool_pose(cube_xy, cube_side, direction):
    """Return the tool pose that seats the crossbar against the cube."""
    half = cube_side / 2
    gap = TOOL_CONTACT_GAP_M
    if direction == "push":
        # The crossbar trails the cube, so its far face touches the cube first.
        y = cube_xy[1] - half - gap - TOOL_LENGTH_M / 2
    else:
        # The crossbar hooks the cube, so its near face touches first.
        y = cube_xy[1] + half + gap - CROSSBAR_INSET_M
    x = cube_xy[0] + half + gap + ARM_HALF_WIDTH_M - ARM_CENTER_OFFSET_M
    return [x, y]


def build(reference, task):
    layout = copy.deepcopy(reference)
    table = layout["Table"]
    table_top = table["default_pos"][2] + table["scale"][2] / 2
    cube = copy.deepcopy(reference["Rigid"]["cube"][0])
    cube.update(
        label="cube",
        category="cube",
        category_idx=0,
        default_pos=[0.0, 0.05 if task == TOOL_PUSH_TASK else 0.30, table_top + CUBE_SIDE_M / 2],
        default_ori=[1.0, 0.0, 0.0, 0.0],
        qpos=[1.0, 0.0, 0.0, 0.0],
        rotate_deg=0.0,
        rotate_rand=False,
        scale=[CUBE_SIDE_M / REFERENCE_CUBE_ASSET_SIDE_M] * 3,
        physics={
            "type": "rigid",
            "mass": CUBE_MASS_KG,
            "static_friction": 0.35,
            "dynamic_friction": 0.25,
            "restitution": 0.0,
        },
        visual={"color": [1.0, 0.25, 0.02]},
        relative_plane="Table",
        margin=0.0,
    )
    tool = copy.deepcopy(reference["Rigid"]["cube"][0])
    pose = tool_pose(cube["default_pos"][:2], CUBE_SIDE_M, "push" if task == TOOL_PUSH_TASK else "pull")
    tool.update(
        label="l_tool",
        category="l_tool",
        category_idx=0,
        default_pos=[pose[0], pose[1], table_top + TOOL_HEIGHT_M / 2],
        default_ori=[1.0, 0.0, 0.0, 0.0],
        qpos=[1.0, 0.0, 0.0, 0.0],
        rotate_deg=0.0,
        rotate_rand=False,
        scale=[TOOL_WIDTH_M, TOOL_LENGTH_M, TOOL_HEIGHT_M],
        physics={
            "type": "rigid",
            "mass": TOOL_MASS_KG,
            "static_friction": 0.5,
            "dynamic_friction": 0.4,
            "restitution": 0.0,
        },
        visual={"color": [0.24, 0.26, 0.29]},
        relative_plane="Table",
        margin=0.0,
    )
    target = copy.deepcopy(reference["Geometry"]["checkerboard"][0])
    target.update(
        label="target_zone",
        default_pos=[
            0.0,
            0.30 if task == TOOL_PUSH_TASK else 0.0,
            table_top - 0.00005,
        ],
        default_ori=[1.0, 0.0, 0.0, 0.0],
        qpos=[1.0, 0.0, 0.0, 0.0],
        scale=[TARGET_SIDE_M / NATIVE_TARGET_SIDE_M] * 2 + [0.00625],
        physics={"type": "geometry", "collision": False},
        visual={},
        relative_plane="Table",
        margin=0.0,
    )
    for obj in (cube, tool, target):
        _set_limits(obj)
    layout["Rigid"] = {"cube": [cube], "l_tool": [tool]}
    layout["Geometry"] = {"checkerboard": [target]}
    layout["Table"]["task_variation"] = {
        "version": 1,
        "task": task,
        "cube_side_m": CUBE_SIDE_M,
        "target_side_m": TARGET_SIDE_M,
        "tool_crossbar_inset_m": CROSSBAR_INSET_M,
        "tool_handle_y_m": HANDLE_CENTER_Y_M,
    }
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in (TOOL_PUSH_TASK, TOOL_PULL_TASK):
        path = args.output_dir / f"{task}.json"
        path.write_text(json.dumps(build(reference, task), indent=2) + "\n")
        print(path)


if __name__ == "__main__":
    main()
