"""Push a reachable cube to an unreachable target with the L-shaped tool."""

from env.environment.task_env import TaskEnv

from .tool_manipulation_common import ToolManipulationCommon


class tool_push_cube(ToolManipulationCommon, TaskEnv):
    direction = "push"
