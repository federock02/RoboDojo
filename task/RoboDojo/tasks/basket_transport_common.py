"""Native goal and progress logic for load-sensitive basket transport."""

from __future__ import annotations

from env.reward_manager.reward_manager import RewardManager

from .custom_task_progress import CustomTaskProgress


class BasketTransportCommon(CustomTaskProgress):
    """Base mixin for native RoboDojo basket transport variants."""

    transport_label = "basket"
    payload_labels: tuple[str, ...] = ()
    target_description = "the black-and-yellow checkerboard target zone"
    basket_description = "orange shopping basket"

    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        # This leaves enough time for an unload / move / reload strategy.
        self.step_lim = 1_800

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def _basket_at_target(self):
        return self.reward_manager.is_AB_xy_distance_within_threshold(
            label_A="basket", label_B="target_zone", threshold=0.10
        )

    def _basket_upright(self):
        return self.reward_manager.is_axis_up(label="basket", axis=[0, 0, 1], threshold=25)

    def _payload_reloaded(self):
        return [self.reward_manager.is_A_in_B(label_A=payload, label_B="basket") for payload in self.payload_labels]

    def _final_checks(self):
        return [
            self._basket_at_target(),
            self._basket_upright(),
            *self._payload_reloaded(),
            self.reward_manager.is_all_gripper_open(open_threshold=0.8),
            self._arms_home_check(),
        ]

    def run_reward(self):
        self._start_custom_progress()
        self.reward_manager.final_check(self._final_checks())

    def get_score(self):
        stages = [
            self._advance_tier(0.25, 20),
            self._advance_tier(0.50, 40),
            ("Basket finishes at target", 60, [self._basket_at_target()]),
        ]
        upright = [self._basket_upright()]
        if self.payload_labels:
            stages += [
                ("Basket finishes upright", 70, upright),
                ("All original payloads are inside", 85, self._payload_reloaded()),
            ]
        else:
            stages.append(("Basket finishes upright", 80, upright))
        self._register_custom_score(stages)

    def gen_instruction(self, env_idx):
        instruction = "Move the shopping basket to the marked target zone"
        if self.payload_labels:
            instruction += " with all objects inside"
        return [instruction + ", then return both arms to their starting poses."]
