"""Empty-basket control variant."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class basket_transport_empty(BasketTransportCommon, TaskEnv):
    payload_labels = ()
    basket_description = "orange shopping basket"
