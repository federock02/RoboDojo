"""Check that the custom basket and cube goals require the native home predicate."""

from importlib import import_module
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CustomArmsHomeTest(unittest.TestCase):
    def test_home_uses_native_defaults_only_for_full_success(self):
        reward_module = ModuleType("env.reward_manager.reward_manager")
        reward_module.RewardManager = object
        with patch.dict(sys.modules, {reward_module.__name__: reward_module}):
            basket = import_module("task.RoboDojo.tasks.basket_transport_common").BasketTransportCommon
            cube = import_module("task.RoboDojo.tasks.cube_manipulation_common").CubeManipulationCommon
        native_home = ("all_robot_back_to_origin", {"pos_threshold": 0.15, "rot_threshold": 20})

        for task_class in (basket, cube):
            with self.subTest(task=task_class.__name__):
                task = task_class.__new__(task_class)
                task._start_custom_progress = lambda: None
                manager = SimpleNamespace(
                    all_robot_back_to_origin=Mock(return_value=native_home),
                    final_check=Mock(),
                    is_AB_xy_distance_within_threshold=Mock(return_value=("at_target", {})),
                    is_axis_up=Mock(return_value=("upright", {})),
                    is_all_gripper_open=Mock(return_value=("grippers_open", {})),
                )
                task.reward_manager = manager
                task.run_reward()
                manager.all_robot_back_to_origin.assert_called_once_with()
                self.assertIn(native_home, manager.final_check.call_args.args[0])
                task.get_score()
                self.assertTrue(all(native_home not in checks for _, _, checks in task._progress_tiers))


if __name__ == "__main__":
    unittest.main()
