"""Pull an unreachable cube to a reachable target with the L-shaped tool."""

from env.environment.task_env import TaskEnv

from .tool_manipulation_common import ToolManipulationCommon


class tool_pull_cube(ToolManipulationCommon, TaskEnv):
    direction = "pull"
    unreachable_object = "cube"
