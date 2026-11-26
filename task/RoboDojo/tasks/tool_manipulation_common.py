"""Native goal, transport milestones and no-lift rule for both L-tool tasks."""

from __future__ import annotations

from env.reward_manager.reward_manager import RewardManager

from .custom_task_progress import CustomTaskProgress

CUBE_LABEL = "cube"
TARGET_LABEL = "target_zone"
TARGET_RADIUS_M = 0.035
MAX_CUBE_LIFT_M = 0.02


class ToolManipulationCommon(CustomTaskProgress):
    """Base mixin for the push/pull tool tasks."""

    direction = ""
    unreachable_object = "target"
    tool_description = "long L-shaped tool"
    target_description = "marked target zone"
    acceptance_radius_m = TARGET_RADIUS_M
    transport_label = CUBE_LABEL
    auxiliary_label = "l_tool"
    max_lift_m = MAX_CUBE_LIFT_M

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
        self._start_custom_progress()
        self.reward_manager.final_check([self._constraints_check(), *self._final_checks()])

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
            "- 20 or 40: Reduce the closest distance to the target by at least 25% "
            "or 50% of the initial distance, respectively, with at least 1 cm of progress.\n"
            "- 75: Also finish with the cube inside the target zone.\n"
            "- 100: Finish with the cube inside the target zone and both arms at home.\n"
            "- 0: The cube rose more than 2 cm above its settled starting height, "
            "or no progress milestone was met."
        )
