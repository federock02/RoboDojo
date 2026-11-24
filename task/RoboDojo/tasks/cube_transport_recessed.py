"""Transport the large cube from a shallow recess to the marked zone."""

import numpy as np

from .cube_manipulation_geometry import arrangement_checks
from .cube_transport_large import cube_transport_large


class cube_transport_recessed(cube_transport_large):
    def _table_top(self, env_idx, large_side):
        position, _ = self._recessed_pose(env_idx, "target_zone")
        return float(position[2]) + self._marker_to_surface(env_idx)

    def _variation(self, env_idx):
        layout = self.scene_manager.layout_manager
        saved = getattr(layout, "saved_layouts", [None] * self.num_envs)[env_idx]
        return (saved or {}).get("Table", {}).get("task_variation", {})

    def _marker_to_surface(self, env_idx):
        # Generated flat markers sit just above the surface. The original
        # marker has thickness and its origin sits just below the surface.
        seeded = self._variation(env_idx).get("task") == "cube_transport_recessed"
        return -0.00005 if seeded else 0.00005

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
            variation = self._variation(env_idx)
            checks = arrangement_checks(
                {"large_cube": self._recessed_pose(env_idx, "large_cube")},
                target_pose,
                float(target_pose[0][2]) + self._marker_to_surface(env_idx),
                stacked=False,
                sides=variation.get("cube_sides") or None,
                target_side=variation.get("target_side", 0.26),
                target_shape=variation.get("target_shape", "square"),
            )
        except (IndexError, ValueError):
            return 0.0
        return float(checks[args["stage"]])
