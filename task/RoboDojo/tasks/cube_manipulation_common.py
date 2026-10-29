"""Native RoboDojo integration for result-only cube placement tasks."""

import numpy as np

from env.reward_manager.reward_manager import RewardManager

from .cube_manipulation_geometry import CUBE_SPECS, arrangement_checks


class CubeManipulationCommon:
    stacked = False

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

    def _check_stage(self, args):
        env_idx = args["env_idx"]
        layout = self.scene_manager.layout_manager

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float).reshape(-1)

        def pose(label):
            position, quaternion = layout.get_instance_pose(env_idx=env_idx, label=label)
            if position is None or quaternion is None:
                raise ValueError(f"missing scene pose: {label}")
            return array(position), array(quaternion)

        labels = CUBE_SPECS if self.stacked else ("large_cube",)
        poses = {label: pose(label) for label in labels}
        name = layout.get_instance_name(env_idx=env_idx, label="large_cube")
        obj = layout.get_scene_object(env_idx, name)
        # The replay starts every cube resting on the tabletop. default_pos is
        # env-relative, as are get_instance_pose results (including tiled envs).
        table_top = float(obj.default_pos[2]) - CUBE_SPECS["large_cube"]["side"] / 2
        try:
            checks = arrangement_checks(poses, pose("target_zone"), table_top, stacked=self.stacked)
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
                (
                    "Stack the three cubes in the black-and-yellow checkerboard target zone, "
                    "with the large orange cube at the bottom, the blue cube in the middle, "
                    "and the small green cube on top."
                )
            ]
        return ["Move the large orange cube to the black-and-yellow checkerboard target zone."]
