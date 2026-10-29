"""Move the large base cube and stack the two smaller cubes on it."""

from env.environment.task_env import TaskEnv

from .cube_manipulation_common import CubeManipulationCommon


class cube_stack_sizes(CubeManipulationCommon, TaskEnv):
    stacked = True
