"""Empty basket seated in a tabletop pocket that blocks sliding."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


class recessed_basket(BasketTransportCommon, TaskEnv):
    payload_labels = ()
