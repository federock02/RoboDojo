"""Shared goal logic for the L-shaped-tool push and pull tasks.

Both variants judge only the final physical state: the cube is inside the
marked target zone, it was never lifted off the tabletop, and both arms have
returned to their initial poses.  The tool itself is an ordinary rigid body;
no privileged intermediate state or hidden tool-attachment check is used.
"""

from __future__ import annotations

from env.reward_manager.reward_manager import RewardManager

CUBE_LABEL = "cube"
TARGET_LABEL = "target_zone"
TARGET_RADIUS_M = 0.035
MAX_CUBE_LIFT_M = 0.02


class ToolManipulationCommon:
    """Base mixin for the push/pull tool tasks."""

    direction = ""
    unreachable_object = "target"
    tool_description = "long L-shaped tool"
    target_description = "marked target zone"
    acceptance_radius_m = TARGET_RADIUS_M
    max_cube_lift_m = MAX_CUBE_LIFT_M

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 900

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
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

    def run_reward(self):
        # Lifting the cube bypasses the intended tool-mediated transport.
        self.reward_manager.query(
            [self.reward_manager.is_lift(label=CUBE_LABEL, z_threshold=self.max_cube_lift_m)],
            0,
        )
        self.reward_manager.final_check(self._final_checks())

    def get_score(self):
        self.reward_manager.final_score([self._final_checks()], [100])

    def gen_instruction(self, env_idx):
        return [
            f"Use the {self.tool_description} to {self.direction} the small cube "
            f"onto the {self.target_description} without lifting the cube."
        ]

    def gen_detailed_instruction(self, env_idx):
        return (
            "## Description\n\n"
            f"Use the {self.tool_description} to {self.direction} the small cube "
            f"onto the {self.target_description}. The cube starts and must finish "
            f"on the tabletop: pick up the tool, not the cube. The {self.unreachable_object} "
            "is beyond the arms' reach in this variant, so the cube has to be moved with the "
            "tool. Keep the cube flat on the table, correct its position until it "
            "rests inside the marked zone, release the tool, and return both arms "
            "to their initial poses.\n\n"
            "## Scoring\n\n"
            "- 0: The cube is not inside the target zone, it was lifted above the "
            "allowed height, or the arms did not return to their initial poses.\n"
            "- 100: The cube is inside the target zone, was never lifted, and both "
            "arms returned to their initial poses."
        )
