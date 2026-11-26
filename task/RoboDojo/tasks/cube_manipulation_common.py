"""Native RoboDojo scoring and diagnostics for cube placement tasks."""

import numpy as np

from env.reward_manager.reward_manager import RewardManager

from .cube_manipulation_geometry import (
    CONTACT_GAP_M,
    CUBE_SPECS,
    EDGE_TOLERANCE_M,
    PENETRATION_M,
    SUPPORT_TOLERANCE_DEG,
    UPRIGHT_TOLERANCE_DEG,
    angle_degrees,
    arrangement_checks,
    footprint_radius,
    rotation_matrix,
    stack_contact,
    support_face,
    target_contains_cube,
    target_local_footprint,
    upright_cube,
)
from .custom_task_progress import CustomTaskProgress


class CubeManipulationCommon(CustomTaskProgress):
    transport_label = "large_cube"
    stacked = False
    upright = False

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        parser = self.reward_manager.func_parser
        parser.cube_arrangement_stage = self._check_stage
        parser.cube_target_fit = self._cube_target_fit
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
        sides = variation.get("cube_sides") or {label: spec["side"] for label, spec in CUBE_SPECS.items()}
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

    def physical_diagnostic_measurements(self, env_idx):
        """Return terminal geometry checks as measured values and thresholds."""
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

        def measurement(name, label, comparison, required, unit, read):
            return {
                "id": f"cube.{name}",
                "label": label,
                "comparison": comparison,
                "required": required,
                "unit": unit,
                "measure": read,
            }

        large_side = sides["large_cube"]
        yield measurement(
            "target_footprint",
            "Target radius needed to fit large cube",
            "<=",
            target_side / 2 + EDGE_TOLERANCE_M,
            "m",
            lambda: check(
                lambda large, target: footprint_radius(target_local_footprint(large, large_side, target), target_shape),
                "large_cube",
                "target_zone",
            ),
        )
        if self.upright:
            yield measurement(
                "upright",
                "Large cube top-face tilt",
                "<=",
                UPRIGHT_TOLERANCE_DEG,
                "deg",
                lambda: check(
                    lambda large: angle_degrees(rotation_matrix(large[1])[2, 2]),
                    "large_cube",
                ),
            )

        supports = [("large_cube", None)]
        if self.stacked:
            supports += [("medium_cube", "large_cube"), ("small_cube", "medium_cube")]

        def contact(upper, lower):
            a = pose(upper)
            b = pose(lower) if lower else None
            if a is None or (lower and b is None):
                return {}
            face, up = support_face(a, sides[upper])
            values = {"tilt": angle_degrees(up)}
            if lower:
                gaps, lateral, lower_up = stack_contact(face, b, sides[lower])
                values.update(
                    support_tilt=angle_degrees(lower_up),
                    extent=float(np.max(np.abs(lateral))),
                )
            else:
                gaps = face[:, 2] - self._table_top(env_idx, sides[upper])
            return dict(values, gap=float(max(gaps)), lowest=float(min(gaps)))

        for upper, lower in supports:
            label = upper.replace("_", " ").capitalize()
            destination = lower.replace("_", " ") if lower else "table"
            specs = [
                (
                    "tilt",
                    f"{label} resting-face tilt",
                    "<=",
                    SUPPORT_TOLERANCE_DEG,
                    "deg",
                ),
                (
                    "gap",
                    f"{label} largest gap above {destination}",
                    "<=",
                    CONTACT_GAP_M,
                    "m",
                ),
                (
                    "lowest",
                    f"{label} lowest gap above {destination}",
                    ">=",
                    -PENETRATION_M,
                    "m",
                ),
            ]
            if lower:
                specs += [
                    (
                        "support_tilt",
                        f"{destination.capitalize()} supporting-face tilt",
                        "<=",
                        SUPPORT_TOLERANCE_DEG,
                        "deg",
                    ),
                    (
                        "extent",
                        f"{label} footprint radius on {destination}",
                        "<=",
                        sides[lower] / 2 + EDGE_TOLERANCE_M,
                        "m",
                    ),
                ]
            for key, title, comparison, required, unit in specs:
                yield measurement(
                    f"{upper}.{key}",
                    title,
                    comparison,
                    required,
                    unit,
                    lambda upper=upper, lower=lower, key=key: contact(upper, lower).get(key),
                )

    def _cube_target_fit(self, args):
        env_idx = args["env_idx"]
        sides, target_side, target_shape = self._geometry(env_idx)
        return float(
            target_contains_cube(
                self._pose(env_idx, "large_cube"),
                sides["large_cube"],
                self._pose(env_idx, "target_zone"),
                target_side,
                target_shape,
            )
        )

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
        self._start_custom_progress()
        self.reward_manager.final_check([self._constraints_check(), self._stage(2 if self.stacked else 0)])

    def get_score(self):
        stages = [self._advance(0.25), self._advance(0.50)]
        if self.stacked:
            self._register_custom_score([*stages, [self._stage(0)], [self._stage(1)]], [15, 30, 50, 75])
        else:
            self._register_custom_score([*stages, [("cube_target_fit", {})]], [25, 50, 75])

    def gen_instruction(self, env_idx):
        if self.stacked:
            return [
                "Stack the three cubes in the marked target zone, with the largest cube at the bottom, the medium cube in the middle, and the smallest cube on top."
            ]
        return ["Move the large cube to the marked target zone."]
