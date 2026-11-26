"""Physics-step measurements and native milestone scoring for custom tasks."""

from dataclasses import dataclass

import numpy as np


def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float).reshape(-1)


@dataclass
class TransportHistory:
    start_z: float
    initial_distance: float
    closest_distance: float
    peak_rise: float = 0.0
    peak_tilt: float = 0.0
    samples: int = 0

    def sample(self, distance, height, tilt):
        self.closest_distance = min(self.closest_distance, distance)
        self.peak_rise = max(self.peak_rise, height - self.start_z)
        self.peak_tilt = max(self.peak_tilt, tilt)
        self.samples += 1

    @property
    def advance(self):
        if self.initial_distance == 0:
            return 0.0
        return (self.initial_distance - self.closest_distance) / self.initial_distance


class CustomTaskProgress:
    transport_label = ""
    max_lift_m = None
    max_tilt_deg = None
    auxiliary_label = None

    def reset(self, seed=None, options=None):
        self._transport_history = {}
        return super().reset(seed=seed, options=options)

    def _transport_pose(self, env_idx, label):
        layout = self.scene_manager.layout_manager
        position, quaternion = layout.get_instance_pose(env_idx=env_idx, label=label)
        position, quaternion = as_array(position), as_array(quaternion)
        norm = np.linalg.norm(quaternion)
        if (
            position.size != 3
            or quaternion.size != 4
            or not np.isfinite(position).all()
            or not np.isfinite(norm)
            or norm <= 1e-12
        ):
            raise ValueError(f"Invalid pose for {label} in environment {env_idx}")
        return position, quaternion / norm

    def _start_custom_progress(self):
        self._transport_targets = {}
        self._transport_history = {}
        self._auxiliary_motion = {}
        for env_idx in self.get_running_env_idx_list():
            target, _ = self._transport_pose(env_idx, "target_zone")
            position, quaternion = self._transport_pose(env_idx, self.transport_label)
            self._transport_targets[env_idx] = target[:2].copy()
            distance = float(np.linalg.norm(position[:2] - target[:2]))
            history = TransportHistory(float(position[2]), distance, distance)
            history.peak_tilt = self._transport_tilt(quaternion)
            self._transport_history[env_idx] = history
            if self.auxiliary_label:
                origin, _ = self._transport_pose(env_idx, self.auxiliary_label)
                self._auxiliary_motion[env_idx] = (origin.copy(), 0.0, 0.0)
        parser = self.reward_manager.func_parser
        parser.custom_transport_milestone = self._custom_transport_milestone
        parser.custom_transport_constraints = self._custom_transport_constraints

    @staticmethod
    def _transport_tilt(quaternion):
        _, x, y, _ = quaternion
        return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))

    def sample_custom_progress(self, env_idx_list):
        for env_idx in env_idx_list:
            history = self._transport_history.get(env_idx)
            if history is None:
                continue
            position, quaternion = self._transport_pose(env_idx, self.transport_label)
            distance = float(np.linalg.norm(position[:2] - self._transport_targets[env_idx]))
            history.sample(distance, float(position[2]), self._transport_tilt(quaternion))
            if self.auxiliary_label:
                origin, peak_displacement, peak_rise = self._auxiliary_motion[env_idx]
                position, _ = self._transport_pose(env_idx, self.auxiliary_label)
                delta = position - origin
                self._auxiliary_motion[env_idx] = (
                    origin,
                    max(peak_displacement, float(np.linalg.norm(delta))),
                    max(peak_rise, float(delta[2])),
                )

    def _custom_transport_milestone(self, args):
        history = self._transport_history[args["env_idx"]]
        return float(
            history.advance >= args["fraction"] and history.initial_distance - history.closest_distance >= 0.01
        )

    def _custom_transport_constraints(self, args):
        history = self._transport_history.get(args["env_idx"])
        return float(
            history is not None
            and (self.max_lift_m is None or history.peak_rise <= self.max_lift_m)
            and (self.max_tilt_deg is None or history.peak_tilt <= self.max_tilt_deg)
        )

    @staticmethod
    def _constraints_check():
        return ("custom_transport_constraints", {})

    def _arms_home_check(self):
        return self.reward_manager.all_robot_back_to_origin()

    def _register_custom_score(self, stages):
        self._progress_tiers = stages

    def custom_episode_score(self, env_idx, succeeded):
        if succeeded:
            return 1.0
        if not self.reward_manager.check_once(self._constraints_check(), env_idx):
            return 0.0
        # Re-evaluate terminal predicates only when scoring this result.
        # Native batch evaluation can inspect rewards while other envs run.
        score = 0.0
        for _, points, checks in self._progress_tiers:
            if not all(self.reward_manager.check_once(check, env_idx) for check in checks):
                break
            score = points / 100.0
        return score

    @staticmethod
    def _advance_tier(fraction, points):
        return (
            f"Reduce target distance by {fraction:.0%} and at least 1 cm",
            points,
            [("custom_transport_milestone", {"fraction": fraction})],
        )

    def custom_progress_report(self, env_idx, *, score=None):
        history = self._transport_history.get(env_idx)
        if history is None:
            return None
        report = {
            "physics_steps_sampled": history.samples,
            "initial_target_distance_m": history.initial_distance,
            "closest_target_distance_m": history.closest_distance,
            "best_target_distance_reduction_fraction": history.advance,
            "peak_vertical_rise_m": history.peak_rise,
            "peak_top_face_tilt_deg": history.peak_tilt,
            "max_lift_m": self.max_lift_m,
            "max_tilt_deg": self.max_tilt_deg,
        }

        if score is not None:
            report["credited_milestones"] = [
                {"label": label, "points": points, "credited": score * 100 >= points}
                for label, points, _ in self._progress_tiers
            ]
        if self.auxiliary_label:
            _, displacement, rise = self._auxiliary_motion[env_idx]
            report["auxiliary_object"] = self.auxiliary_label
            report["auxiliary_peak_displacement_m"] = displacement
            report["auxiliary_peak_vertical_rise_m"] = rise
        return report
