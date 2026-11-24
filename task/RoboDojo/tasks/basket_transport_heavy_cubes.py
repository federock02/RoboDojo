"""Large removable cubes using RoboDojo's default gripper control."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_heavy_cubes(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1")
