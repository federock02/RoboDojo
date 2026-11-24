"""Native RoboDojo integration for result-only cube placement tasks."""

import numpy as np
from env.reward_manager.reward_manager import RewardManager

from .cube_manipulation_geometry import (
    CUBE_SPECS,
    arrangement_checks,
    supported_on_cube,
    supported_on_table,
    target_contains_cube,
    upright_cube,
)


class CubeManipulationCommon:
    stacked = False
    upright = False

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.reward_manager.func_parser.cube_arrangement_stage = self._check_stage
        self.step_lim = 1200

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _pose(self, env_idx, label):
        layout = self.scene_manager.layout_manager
        position, quaternion = layout.get_instance_pose(env_idx=env_idx, label=label)
        if position is None or quaternion is None:
            raise ValueError(f"missing scene pose: {label}")

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float).reshape(-1)

        return array(position), array(quaternion)

    def _geometry(self, env_idx):
        layout = self.scene_manager.layout_manager
        saved = getattr(layout, "saved_layouts", [None] * self.num_envs)[env_idx]
        variation = (saved or {}).get("Table", {}).get("task_variation", {})
        sides = variation.get("cube_sides") or {
            label: spec["side"] for label, spec in CUBE_SPECS.items()
        }
        return (
            sides,
            variation.get("target_side", 0.26),
            variation.get("target_shape", "square"),
        )

    def _table_top(self, env_idx, large_side):
        layout = self.scene_manager.layout_manager
        name = layout.get_instance_name(env_idx=env_idx, label="large_cube")
        obj = layout.get_scene_object(env_idx, name)
        # Saved and measured coordinates are both environment-relative.
        return float(obj.default_pos[2]) - large_side / 2

    def cube_diagnostic_predicates(self, env_idx):
        """Expose independent read-only predicates without registering rewards."""
        sides, target_side, target_shape = self._geometry(env_idx)
        layout = self.scene_manager.layout_manager

        def pose(label):
            if layout.get_instance_name(env_idx=env_idx, label=label) is None:
                return None
            position, quaternion = self._pose(env_idx, label)
            if (
                not np.isfinite(position).all()
                or not np.isfinite(quaternion).all()
                or np.linalg.norm(quaternion) < 1e-12
            ):
                raise ValueError(f"invalid scene pose: {label}")
            return position, quaternion

        def check(predicate, *labels):
            poses = [pose(label) for label in labels]
            return None if any(value is None for value in poses) else predicate(*poses)

        predicates = {
            "target_footprint": lambda: check(
                lambda large, target: target_contains_cube(
                    large, sides["large_cube"], target, target_side, target_shape
                ),
                "large_cube",
                "target_zone",
            ),
            "table_support": lambda: check(
                lambda large: supported_on_table(
                    large,
                    sides["large_cube"],
                    self._table_top(env_idx, sides["large_cube"]),
                ),
                "large_cube",
            ),
        }
        if self.upright:
            predicates["upright"] = lambda: check(upright_cube, "large_cube")
        if self.stacked:
            for upper, lower in (
                ("medium_cube", "large_cube"),
                ("small_cube", "medium_cube"),
            ):
                predicates[upper + "_support"] = lambda upper=upper, lower=lower: check(
                    lambda a, b: supported_on_cube(a, sides[upper], b, sides[lower]),
                    upper,
                    lower,
                )
        return predicates

    def _check_stage(self, args):
        env_idx = args["env_idx"]
        labels = CUBE_SPECS if self.stacked else ("large_cube",)
        poses = {label: self._pose(env_idx, label) for label in labels}
        sides, target_side, target_shape = self._geometry(env_idx)
        table_top = self._table_top(env_idx, sides["large_cube"])
        try:
            checks = arrangement_checks(
                poses,
                self._pose(env_idx, "target_zone"),
                table_top,
                stacked=self.stacked,
                sides=sides,
                target_side=target_side,
                target_shape=target_shape,
            )
            if self.upright and not upright_cube(poses["large_cube"]):
                return 0.0
        except ValueError:
            return 0.0
        return float(checks[args["stage"]])

    def _stage(self, index):
        return ("cube_arrangement_stage", {"stage": index})

    def run_reward(self):
        self.reward_manager.final_check([self._stage(2 if self.stacked else 0)])

    def get_score(self):
        # Register final-state scores rather than execution milestones.
        count = 3 if self.stacked else 1
        self.reward_manager.final_score(
            [[self._stage(index)] for index in range(count)],
            [30, 65, 100] if self.stacked else [100],
            score_mode="by_count",
        )

    def gen_instruction(self, env_idx):
        if self.stacked:
            return [
                "Stack the three cubes in the marked target zone, with the largest cube at the bottom, the medium cube in the middle, and the smallest cube on top."
            ]
        return ["Move the large cube to the marked target zone."]
