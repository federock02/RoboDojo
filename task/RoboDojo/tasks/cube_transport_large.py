"""Move a light cube wider than a single X5 gripper."""

from env.environment.task_env import TaskEnv

from .cube_manipulation_common import CubeManipulationCommon


class cube_transport_large(CubeManipulationCommon, TaskEnv):
    pass
