"""Heavily loaded basket on a high-friction tabletop that resists sliding."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_heavy_cubes_high_friction(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1")
