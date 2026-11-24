"""Large cubes; payload0 is physically fixed to the basket, not the world."""

from env.environment.task_env import TaskEnv

from .basket_transport_common import BasketTransportCommon


def attach_cube(
    stage,
    basket_path: str,
    cube_path: str,
    joint_path: str,
    *,
    basket_frame=None,
    cube_frame=None,
):
    """Lock the current relative pose with an unbreakable USD fixed joint."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    basket = stage.GetPrimAtPath(basket_path)
    cube = stage.GetPrimAtPath(cube_path)
    if not basket or not cube or not basket.HasAPI(UsdPhysics.RigidBodyAPI) or not cube.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("fixed payload requires two valid rigid bodies")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    # Strip scale: joint frames use rigid-body coordinates, not mesh scale.
    if (basket_frame is None) != (cube_frame is None):
        raise ValueError("supply both rigid-body frames or neither")
    basket_world = (
        basket_frame if basket_frame is not None else cache.GetLocalToWorldTransform(basket).RemoveScaleShear()
    )
    cube_world = cube_frame if cube_frame is not None else cache.GetLocalToWorldTransform(cube).RemoveScaleShear()
    relative = cube_world * basket_world.GetInverse()
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(basket_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(relative.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.CreateBreakForceAttr().Set(float("inf"))
    joint.CreateBreakTorqueAttr().Set(float("inf"))
    return joint


class basket_transport_fixed_cube(BasketTransportCommon, TaskEnv):
    payload_labels = ("payload0", "payload1", "payload2")

    def reset(self, seed=None, options=None):
        import omni.usd
        from pxr import Gf

        stage = omni.usd.get_context().get_stage()
        # Old scene objects are replaced on reset; remove old constraints first.
        for path in getattr(self, "_payload_joint_paths", ()):
            if stage is not None:
                stage.RemovePrim(path)
        super().reset(seed=seed, options=options)
        for robot in self.robot_manager.robot_list:
            if robot.type == "target":
                robot.gripper_target_step_fraction = 1.0
                robot.interpolate_gripper_targets = False
        # The first reset launches Isaac and may replace the startup stage.
        stage = omni.usd.get_context().get_stage()
        layout = self.scene_manager.layout_manager
        self._payload_joint_paths = []
        for env_idx in range(self.num_envs):
            objects = []
            for label in ("basket", "payload0"):
                name = layout.get_instance_name(env_idx=env_idx, label=label)
                obj = layout.get_scene_object(env_idx, name)
                if obj is None:
                    raise ValueError(f"missing fixed-payload object: {label}")
                objects.append(obj)
            basket, cube = objects
            path = str(basket.prim_path) + "/fixed_payload_joint"

            # EvalEnv.setup_scene restores saved poses AFTER this task reset.
            # Bodies may currently be falling while support fixtures are reset;
            # those transient poses must never define the permanent attachment.
            def saved_frame(obj):
                w, x, y, z = obj.default_ori
                return Gf.Matrix4d(
                    Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))),
                    Gf.Vec3d(*obj.default_pos),
                )

            attach_cube(
                stage,
                basket.prim_path,
                cube.prim_path,
                path,
                basket_frame=saved_frame(basket),
                cube_frame=saved_frame(cube),
            )
            self._payload_joint_paths.append(path)
