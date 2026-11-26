"""Regression tests for the L-tool evaluator without an Isaac Sim dependency."""

from importlib import import_module
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class LToolTransportTest(unittest.TestCase):
    def test_full_credit_requires_tool_progress_and_allows_lifting(self):
        module = ModuleType("env.reward_manager.reward_manager")
        module.RewardManager = object
        with patch.dict(sys.modules, {module.__name__: module}):
            task_class = import_module("task.RoboDojo.tasks.tool_manipulation_common").ToolManipulationCommon

        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                task = task_class.__new__(task_class)
                poses = {"cube": [0.0, 0.0, 0.0], "target_zone": [0.3, 0.0, 0.0], "l_tool": [0, 0, 0]}
                impulse = np.zeros((1, 1, 3))
                task.get_running_env_idx_list = lambda: [0]
                task._transport_pose = lambda _, label: (np.array(poses[label]), np.array([1, 0, 0, 0]))
                task._tool_contact_view = lambda _: SimpleNamespace(get_contact_force_matrix=lambda _: impulse)
                task._final_checks = lambda: [("on_target", {}), ("arms_home", {})]
                registered = []
                task.reward_manager = SimpleNamespace(func_parser=SimpleNamespace(), final_check=registered.extend)
                task.run_reward()

                def checks_pass():
                    parser = task.reward_manager.func_parser
                    return all(getattr(parser, name)({**args, "env_idx": 0}) for name, args in registered)

                task.reward_manager.func_parser.on_target = lambda _: abs(poses["cube"][0] - 0.3) < 0.035
                task.reward_manager.func_parser.arms_home = lambda _: True
                if hybrid:
                    poses["cube"] = [0.10, 0.0, 0.07]
                    task.sample_custom_progress([0])
                impulse[0, 0, 0] = 0.05
                task.sample_custom_progress([0])  # Brief contact without translation.
                impulse[:] = 0.0
                poses["cube"] = [0.3, 0.0, 0.0]
                task.sample_custom_progress([0])
                self.assertFalse(checks_pass())

                # A fresh run retains neither contact nor movement from the first run.
                poses["cube"] = [0.0, 0.0, 0.0]
                registered.clear()
                task.run_reward()
                if hybrid:
                    poses["cube"] = [0.10, 0.0, 0.07]
                    task.sample_custom_progress([0])
                impulse[0, 0, 0] = 0.05
                poses["cube"] = [0.3, 0.0, 0.0]
                task.sample_custom_progress([0])
                self.assertTrue(checks_pass())
                self.assertEqual(task.custom_episode_score(0, checks_pass()), 1.0)
                report = task.custom_progress_report(0)
                self.assertEqual(report["peak_vertical_rise_m"] > 0.02, hybrid)
                self.assertIsNone(report["max_lift_m"])


if __name__ == "__main__":
    unittest.main()
