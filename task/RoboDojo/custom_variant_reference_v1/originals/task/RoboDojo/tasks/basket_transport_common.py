"""Shared goal logic for the load-sensitive basket transport tasks.

The scene configuration supplies the variant-specific payload count and mass.
The task itself deliberately only judges the final physical state: basket on
the target, upright, and all payloads back in the basket.  It does not encode
an unloading plan as a privileged intermediate reward.
"""

from __future__ import annotations

from env.reward_manager.reward_manager import RewardManager


class BasketTransportCommon:
    """Base mixin for native RoboDojo basket transport variants."""

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
        ]

    def run_reward(self):
        self.reward_manager.final_check(self._final_checks())

    def get_score(self):
        """Score only observable progress toward the final physical state."""
        basket_at_target = self._basket_at_target()
        basket_upright = self._basket_upright()
        payloads_reloaded = self._payload_reloaded()
        self.reward_manager.score(
            [
                [basket_at_target],
                [basket_at_target, basket_upright],
                [basket_at_target, basket_upright, *payloads_reloaded],
                self._final_checks(),
            ],
            [25, 50, 75, 100],
            score_mode="transition",
        )

    def gen_instruction(self, env_idx):
        instruction = f"Move the {self.basket_description} to {self.target_description}"
        if self.payload_labels:
            instruction += " with all objects inside"
        return [instruction + "."]
