"""CPU regression for scorer state that must survive an object's return to goal."""

from types import SimpleNamespace
import unittest

import numpy as np

from task.RoboDojo.tasks.custom_task_progress import CustomTaskProgress, TransportHistory


class ProgressTest(unittest.TestCase):
    def test_tiers_are_nested_and_report_credit_from_native_score(self):
        progress = CustomTaskProgress()
        progress._transport_history = {0: TransportHistory(0.1, 0.4, 0.16)}
        at_target = True

        def check_once(check, env_idx):
            name, args = check
            if name == "at_target":
                return at_target
            return {
                "custom_transport_constraints": progress._custom_transport_constraints,
                "custom_transport_milestone": progress._custom_transport_milestone,
            }[name]({**args, "env_idx": env_idx})

        progress.reward_manager = SimpleNamespace(check_once=check_once)
        progress._register_custom_score(
            [
                progress._advance_tier(0.25, 20),
                progress._advance_tier(0.50, 40),
                ("Finish at target", 75, [("at_target", {})]),
            ]
        )
        self.assertEqual(progress.custom_episode_score(0, False), 0.75)
        at_target = False
        score = progress.custom_episode_score(0, False)
        self.assertEqual(score, 0.40)
        stages = progress.custom_progress_report(0, score=score)["credited_milestones"]
        self.assertEqual([stage["credited"] for stage in stages], [True, True, False])
        self.assertEqual(progress.custom_episode_score(0, True), 1.0)

    def test_path_violation_survives_terminal_recovery_and_clears_on_reset(self):
        for axis in ("lift", "tilt"):
            with self.subTest(axis=axis):
                progress = CustomTaskProgress()
                progress.transport_label = "cube"
                if axis == "lift":
                    progress.max_lift_m = 0.02
                else:
                    progress.max_tilt_deg = 10.0
                origin = [np.zeros(3), np.array([1.0, 0, 0, 0])]
                target = [np.array([0.3, 0, 0]), origin[1]]
                current = origin.copy()
                progress._transport_pose = lambda _, label: current if label == "cube" else target
                progress.get_running_env_idx_list = lambda: [0, 1]
                progress.reward_manager = SimpleNamespace(func_parser=SimpleNamespace())
                progress._start_custom_progress()
                args = {"env_idx": 0}
                self.assertFalse(progress._custom_transport_milestone(args | {"fraction": 0.25}))
                current[0] = np.array([0.16, 0, 0.03 if axis == "lift" else 0.0])
                if axis == "tilt":
                    current[1] = np.array([2**-0.5, 2**-0.5, 0, 0])
                progress.sample_custom_progress([0])
                current[:] = target
                progress.sample_custom_progress([0])

                self.assertFalse(progress._custom_transport_constraints(args))
                self.assertTrue(progress._custom_transport_milestone(args | {"fraction": 0.5}))
                self.assertTrue(progress._custom_transport_constraints({"env_idx": 1}))
                self.assertEqual(progress.custom_progress_report(0)["physics_steps_sampled"], 2)
                self.assertEqual(progress.custom_progress_report(1)["physics_steps_sampled"], 0)
                current[:] = origin
                progress._start_custom_progress()
                self.assertTrue(progress._custom_transport_constraints(args))
                self.assertEqual(progress.custom_progress_report(0)["physics_steps_sampled"], 0)


if __name__ == "__main__":
    unittest.main()
