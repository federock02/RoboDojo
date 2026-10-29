"""Scene dimensions and final-state geometry; no execution-history requirements."""

from itertools import product

import numpy as np

CUBE_SPECS = {
    "large_cube": {
        "side": 0.20,
        "mass": 0.4,
        "xy": [-0.22, -0.04],
        "color": [1.0, 0.25, 0.02],
    },
    "medium_cube": {
        "side": 0.06,
        "mass": 0.1,
        "xy": [-0.22, -0.26],
        "color": [0.05, 0.25, 0.8],
    },
    "small_cube": {
        "side": 0.035,
        "mass": 0.04,
        "xy": [0.02, -0.26],
        "color": [0.15, 0.75, 0.15],
    },
}
TARGET_XY = [0.22, -0.04]
TARGET_SIDE_M = 0.26
CONTACT_GAP_M = 0.006
PENETRATION_M = 0.003
EDGE_TOLERANCE_M = 0.003


def rotation_matrix(quaternion):
    """Convert a finite, nonzero wxyz quaternion to a rotation matrix."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("invalid cube orientation")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cube_vertices(pose, side):
    position, quaternion = pose
    position = np.asarray(position, dtype=float).reshape(3)
    if not np.isfinite(position).all():
        raise ValueError("invalid cube position")
    local = np.array(list(product((-side / 2, side / 2), repeat=3)))
    return local @ rotation_matrix(quaternion).T + position


def bottom_face(pose, side):
    rotation = rotation_matrix(pose[1])
    vertical_axis = int(np.argmax(np.abs(rotation[2])))
    # A resting cube may use any of its six faces and any yaw.
    if abs(rotation[2, vertical_axis]) < np.cos(np.deg2rad(10)):
        return None
    local = np.array(list(product((-side / 2, side / 2), repeat=3)))
    bottom_sign = -np.sign(rotation[2, vertical_axis])
    local = local[np.sign(local[:, vertical_axis]) == bottom_sign]
    return local @ rotation.T + np.asarray(pose[0]).reshape(3)


def supported_on_table(pose, side, table_top):
    bottom = bottom_face(pose, side)
    if bottom is None:
        return False
    gaps = bottom[:, 2] - table_top
    return bool(np.all((gaps >= -PENETRATION_M) & (gaps <= CONTACT_GAP_M)))


def supported_on_cube(upper, upper_side, lower, lower_side):
    """Require the upper bottom face to lie on and within the lower top face."""
    bottom = bottom_face(upper, upper_side)
    if bottom is None:
        return False
    rotation = rotation_matrix(lower[1])
    vertical_axis = int(np.argmax(np.abs(rotation[2])))
    if abs(rotation[2, vertical_axis]) < np.cos(np.deg2rad(10)):
        return False
    local = (bottom - np.asarray(lower[0]).reshape(3)) @ rotation
    gaps = local[:, vertical_axis] * np.sign(rotation[2, vertical_axis]) - lower_side / 2
    lateral = np.delete(local, vertical_axis, axis=1)
    return bool(
        np.all((gaps >= -PENETRATION_M) & (gaps <= CONTACT_GAP_M))
        and np.all(np.abs(lateral) <= lower_side / 2 + EDGE_TOLERANCE_M)
    )


def arrangement_checks(poses, target_pose, table_top, *, stacked):
    """Return final placement stages; a previous lift never invalidates a result."""
    large = poses["large_cube"]
    target_position, target_quaternion = target_pose
    vertices = cube_vertices(large, CUBE_SPECS["large_cube"]["side"])
    target_local = (vertices - np.asarray(target_position).reshape(3)) @ rotation_matrix(target_quaternion)
    in_target = np.all(np.abs(target_local[:, :2]) <= TARGET_SIDE_M / 2 + EDGE_TOLERANCE_M)
    base = bool(in_target and supported_on_table(large, CUBE_SPECS["large_cube"]["side"], table_top))
    if not stacked:
        return (base,)
    medium = supported_on_cube(poses["medium_cube"], 0.06, large, 0.20)
    small = supported_on_cube(poses["small_cube"], 0.035, poses["medium_cube"], 0.06)
    return (base, bool(base and medium), bool(base and medium and small))
