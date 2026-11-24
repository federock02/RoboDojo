"""Transport scoring at the target height, independent of the starting fixture."""

from .cube_transport_large import cube_transport_large


class CubeTransportFixtureCommon(cube_transport_large):
    def _table_top(self, env_idx, large_side):
        position, _ = self._pose(env_idx, "target_zone")
        layout = self.scene_manager.layout_manager
        saved = getattr(layout, "saved_layouts", [None] * self.num_envs)[env_idx]
        variation = (saved or {}).get("Table", {}).get("task_variation", {})
        # The fixed marker is a thin mesh; the generated marker is a plane.
        return float(position[2]) + (-0.00005 if "seed" in variation else 0.00005)
