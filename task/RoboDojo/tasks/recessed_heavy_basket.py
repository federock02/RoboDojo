"""Recessed basket carrying the two heavy 8 cm cubes used by the heavy variant."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class recessed_heavy_basket(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1")
