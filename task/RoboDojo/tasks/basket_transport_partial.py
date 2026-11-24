"""Partially loaded but still movable basket variant."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_partial(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1")
