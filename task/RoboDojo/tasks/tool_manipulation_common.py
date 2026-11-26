"""Native goal and tool-mediated transport milestones for both L-tool tasks."""

from __future__ import annotations

import numpy as np

from env.reward_manager.reward_manager import RewardManager

from .custom_task_progress import CustomTaskProgress, as_array

CUBE_LABEL = "cube"
TARGET_LABEL = "target_zone"
TARGET_RADIUS_M = 0.035
TOOL_LABEL = "l_tool"
MIN_TOOL_PROGRESS_M = 0.01
CONTACT_IMPULSE_EPSILON_NS = 1e-6


class ToolManipulationCommon(CustomTaskProgress):
    """Base mixin for the push/pull tool tasks."""

    direction = ""
    unreachable_object = "target"
    tool_description = "long L-shaped tool"
    target_description = "marked target zone"
    acceptance_radius_m = TARGET_RADIUS_M
    transport_label = CUBE_LABEL
    auxiliary_label = TOOL_LABEL

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 900

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        self._tool_contacts = {}
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _cube_at_target(self):
        return self.reward_manager.is_AB_xy_distance_within_threshold(
            label_A=CUBE_LABEL,
            label_B=TARGET_LABEL,
            threshold=self.acceptance_radius_m,
        )

    def _final_checks(self):
        return [
            self._cube_at_target(),
            self.reward_manager.all_robot_back_to_origin(),
        ]

    def _tool_contact_view(self, env_idx):
        from pxr import PhysxSchema

        layout = self.scene_manager.layout_manager
        tool = layout.get_scene_object(env_idx, layout.get_instance_name(env_idx, TOOL_LABEL))
        cube = layout.get_scene_object(env_idx, layout.get_instance_name(env_idx, CUBE_LABEL))
        prim = tool.stage.GetPrimAtPath(tool.usd_prim_path)
        if not prim or not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise RuntimeError("Installed L-tool asset lacks contact reporting; install the pinned asset")
        view = tool.physics_sim_view.create_rigid_contact_view(tool.usd_prim_path, filter_patterns=[cube.usd_prim_path])
        if view is None or (view.sensor_count, view.filter_count) != (1, 1):
            raise RuntimeError(f"Cannot monitor the L-tool and cube contact pair in environment {env_idx}")
        return view

    def run_reward(self):
        self._start_custom_progress()
        self._tool_contacts = {index: self._tool_contact_view(index) for index in self._transport_history}
        self._last_cube_distance = {
            index: history.initial_distance for index, history in self._transport_history.items()
        }
        self._tool_progress_m = dict.fromkeys(self._tool_contacts, 0.0)
        self._tool_contact_steps = dict.fromkeys(self._tool_contacts, 0)
        self.reward_manager.func_parser.custom_tool_transport = self._tool_transport_satisfied
        self.reward_manager.final_check(
            [self._constraints_check(), ("custom_tool_transport", {}), *self._final_checks()]
        )

    def sample_custom_progress(self, env_idx_list):
        super().sample_custom_progress(env_idx_list)
        for index in env_idx_list:
            if index not in self._tool_contacts:
                continue
            position, _ = self._transport_pose(index, CUBE_LABEL)
            distance = float(np.linalg.norm(position[:2] - self._transport_targets[index]))
            # A unit conversion interval returns the impulse vector for this step.
            impulse = as_array(self._tool_contacts[index].get_contact_force_matrix(1.0))
            if impulse.size != 3 or not np.isfinite(impulse).all():
                raise RuntimeError(f"Invalid L-tool contact impulse in environment {index}")
            if np.linalg.norm(impulse) > CONTACT_IMPULSE_EPSILON_NS:
                self._tool_contact_steps[index] += 1
                self._tool_progress_m[index] += self._last_cube_distance[index] - distance
            self._last_cube_distance[index] = distance

    def _tool_transport_satisfied(self, args):
        return float(self._tool_progress_m[args["env_idx"]] >= MIN_TOOL_PROGRESS_M)

    def custom_progress_report(self, env_idx, *, score=None):
        report = super().custom_progress_report(env_idx, score=score)
        if report is not None:
            position, _ = self._transport_pose(env_idx, CUBE_LABEL)
            report.update(
                cube_target_distance_m=float(np.linalg.norm(position[:2] - self._transport_targets[env_idx])),
                target_radius_m=self.acceptance_radius_m,
                tool_contact_steps=self._tool_contact_steps[env_idx],
                tool_contact_progress_m=self._tool_progress_m[env_idx],
                min_tool_contact_progress_m=MIN_TOOL_PROGRESS_M,
            )
        return report

    def get_score(self):
        self._register_custom_score(
            [
                self._advance_tier(0.25, 20),
                self._advance_tier(0.50, 40),
                ("Cube finishes at target", 75, [self._cube_at_target()]),
            ],
        )

    def gen_instruction(self, env_idx):
        return [
            f"Use the {self.tool_description} to {self.direction} the small cube "
            f"onto the {self.target_description}, then return both arms home."
        ]

    def gen_detailed_instruction(self, env_idx):
        return (
            "## Description\n\n"
            f"Use the {self.tool_description} to {self.direction} the small cube "
            f"onto the {self.target_description}. You may reposition or lift the cube by hand "
            f"as part of the solution. The {self.unreachable_object} is placed far from the arms. "
            "Use the tool to move the cube toward the target, finish with the cube inside the zone, "
            "and return both arms to their initial poses.\n\n"
            "## Scoring\n\n"
            "- 20 or 40: Reduce the closest distance to the target by at least 25% "
            "or 50% of the initial distance, respectively, with at least 1 cm of progress.\n"
            "- 75: Also finish with the cube inside the target zone.\n"
            "- 100: Finish with the cube inside the target zone and both arms at home, "
            "with at least 1 cm of net target approach during tool–cube contact."
        )
