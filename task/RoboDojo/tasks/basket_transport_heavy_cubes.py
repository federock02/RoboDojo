"""Large removable cubes, using the full native gripper position target."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_heavy_cubes(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        for robot in self.robot_manager.robot_list:
            if robot.type == "target":
                robot.gripper_target_step_fraction = 1.0
                robot.interpolate_gripper_targets = False
