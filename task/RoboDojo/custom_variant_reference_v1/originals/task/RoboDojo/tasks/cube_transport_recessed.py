"""Transport the large cube from a shallow recess to the marked zone."""

import numpy as np

from .cube_manipulation_geometry import arrangement_checks
from .cube_transport_large import cube_transport_large


class cube_transport_recessed(cube_transport_large):
    def _table_top(self, env_idx, large_side):
        position, _ = self._recessed_pose(env_idx, "target_zone")
        # The marker origin is 0.05 mm below the surrounding tabletop.
        return float(position[2]) + 0.00005

    def _recessed_pose(self, env_idx, label):
        layout = self.scene_manager.layout_manager
        position, quaternion = layout.get_instance_pose(env_idx=env_idx, label=label)

        def array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float).reshape(-1)

        return array(position), array(quaternion)

    def _check_stage(self, args):
        # The original shared cube scorer derives table height from the start
        # pose, which is below the target surface in this variant.
        env_idx = args["env_idx"]
        try:
            target_pose = self._recessed_pose(env_idx, "target_zone")
            checks = arrangement_checks(
                {"large_cube": self._recessed_pose(env_idx, "large_cube")},
                target_pose,
                float(target_pose[0][2]) + 0.00005,
                stacked=False,
            )
        except (IndexError, ValueError):
            return 0.0
        return float(checks[args["stage"]])
