"""Overloaded basket variant for cross-episode failure-reasoning studies."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_overloaded(BasketTransportCommon, TaskEnv):
    payload_labels = tuple(f"payload{i}" for i in range(8))
