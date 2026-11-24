"""Versioned, CPU-only seeded layouts for project-owned tasks."""

import itertools
import math
import operator
import random
from copy import deepcopy

BASKET_TASKS = tuple(
    "basket_transport_" + s
    for s in ("empty", "partial", "overloaded", "heavy_cubes", "fixed_cube")
)
RECESSED_BASKET_TASK = "recessed_basket"
RECESSED_HEAVY_BASKET_TASK = "recessed_heavy_basket"
RECESSED_BASKET_TASKS = (RECESSED_BASKET_TASK, RECESSED_HEAVY_BASKET_TASK)
HIGH_FRICTION_EMPTY_BASKET_TASK = "basket_transport_empty_high_friction"
HIGH_FRICTION_PARTIAL_BASKET_TASK = "basket_transport_partial_high_friction"
HIGH_FRICTION_HEAVY_BASKET_TASK = "basket_transport_heavy_cubes_high_friction"
HIGH_FRICTION_BASKET_TASKS = (
    HIGH_FRICTION_EMPTY_BASKET_TASK,
    HIGH_FRICTION_PARTIAL_BASKET_TASK,
    HIGH_FRICTION_HEAVY_BASKET_TASK,
)
BASKET_FAMILY_TASKS = BASKET_TASKS + RECESSED_BASKET_TASKS + HIGH_FRICTION_BASKET_TASKS
PARTIAL_BASKET_TASKS = ("basket_transport_partial", HIGH_FRICTION_PARTIAL_BASKET_TASK)
HEAVY_BASKET_TASKS = (
    "basket_transport_heavy_cubes",
    "basket_transport_fixed_cube",
    RECESSED_HEAVY_BASKET_TASK,
    HIGH_FRICTION_HEAVY_BASKET_TASK,
)
RECESSED_CUBE_TASK = "cube_transport_recessed"
FRAGILE_CUBE_TASK = "cube_transport_recessed_fragile"
DEEP_CUBE_TASK = "cube_transport_recessed_deep"
PEGGED_CUBE_TASK = "cube_transport_pegged"
PINNED_CUBE_TASK = "cube_transport_pinned"
NEW_CUBE_TASKS = (FRAGILE_CUBE_TASK, DEEP_CUBE_TASK, PEGGED_CUBE_TASK, PINNED_CUBE_TASK)
HIGH_FRICTION_CUBE_TASK = "cube_transport_high_friction"
FRICTION_TASKS = (*HIGH_FRICTION_BASKET_TASKS, HIGH_FRICTION_CUBE_TASK)
RECESS_DEPTH_M = 0.015
RECESS_SIDE_CLEARANCE_M = 0.001
# The basket rests on a pocket floor 3 cm below the surrounding tabletop so
# that the four fixed sections block horizontal motion out of the pocket.
RECESSED_BASKET_DEPTH_M = 0.030
BASKET_POCKET_CLEARANCE_M = 0.001
BASKET_SUPPORT_CATEGORY = "basket_support"
RECESS_DEPTHS = {
    RECESSED_CUBE_TASK: RECESS_DEPTH_M,
    FRAGILE_CUBE_TASK: RECESS_DEPTH_M,
    DEEP_CUBE_TASK: 0.060,
}
SPECIAL_CUBE_CATEGORIES = {
    FRAGILE_CUBE_TASK: "cube_fragile",
    PEGGED_CUBE_TASK: "cube_socket",
    PINNED_CUBE_TASK: "cube_through_hole",
}
PEG_CATEGORY = "cube_retaining_peg"
REFERENCE_CUBE_SIDE_M = 0.20
PEG_DIAMETER_M = 0.024
PEG_HEIGHT_M = 0.040
SOCKET_DIAMETER_M = 0.028
SOCKET_DEPTH_M = 0.045
# A static high-friction mat resting on the tabletop. It is raised slightly
# so dynamic objects contact only the mat, never the native table surface
# beneath it. Its friction is baked into the mat's USD because RoboDojo does
# not apply runtime physics materials to evaluation-path objects.
FRICTION_SURFACE_CATEGORY = "high_friction_surface"
SURFACE_STATIC_FRICTION = 2.0
SURFACE_DYNAMIC_FRICTION = 2.0
SURFACE_THICKNESS_M = 0.005
# The pinned cube rests on a static pedestal; a free pin runs through both.
PEDESTAL_CATEGORY = "cube_pedestal"
PEDESTAL_HEIGHT_M = 0.040
PIN_CATEGORY = "cube_pin"
PIN_DIAMETER_M = 0.024
PIN_EXPOSED_M = 0.035
THROUGH_HOLE_DIAMETER_M = 0.028
DEFAULT_TASKS = (
    BASKET_TASKS
    + RECESSED_BASKET_TASKS
    + (
        "cube_transport_large",
        "cube_stack_sizes",
        RECESSED_CUBE_TASK,
    )
)
# New tasks are appended so existing task-to-asset indices stay stable.
TASKS = DEFAULT_TASKS + NEW_CUBE_TASKS + FRICTION_TASKS
CUBE_TASKS = (
    "cube_transport_large",
    "cube_stack_sizes",
    RECESSED_CUBE_TASK,
    *NEW_CUBE_TASKS,
    HIGH_FRICTION_CUBE_TASK,
)
COLORS = (
    (0.95, 0.22, 0.05),
    (0.05, 0.4, 0.9),
    (0.15, 0.75, 0.25),
    (0.8, 0.15, 0.65),
    (0.95, 0.75, 0.05),
    (0.05, 0.75, 0.75),
)
SHAPES = ("square", "round", "cross", "circle")
# Calibrated from basket_grasp asset 00002's aligned bounding box.  Its local
# lower face is 0.1095 m below the asset origin. Keep a small gap so PhysX can
# settle the basket onto the tabletop without starting it visibly suspended.
BASKET_BOTTOM_LOCAL_Z_M = -0.1095
BASKET_TABLE_CLEARANCE_M = 0.003


def target_contains(points, side, shape, tolerance=0.003):
    """Test local XY points against the marked region (ring includes interior)."""
    r = side / 2 + tolerance
    for x, y in points:
        x, y = abs(x), abs(y)
        if shape in ("round", "circle"):
            valid = x * x + y * y <= r * r
        elif shape == "cross":
            # Wide X: rotated union of rectangles. The central square fits the base cube.
            u, v = abs(x + y) / math.sqrt(2), abs(x - y) / math.sqrt(2)
            valid = max(x, y) <= r and max(u, v) <= r and min(u, v) <= r * 0.50
        else:
            valid = max(x, y) <= r
        if not valid:
            return False
    return True


def target_contains_footprint(points, side, shape, tolerance=0.003):
    """Contain a convex footprint, including exact segment tests across X notches."""
    if not target_contains(points, side, shape, tolerance):
        return False
    if shape != "cross":
        return True  # Disks and squares are convex.
    r = side / 2 + tolerance
    boundaries = (-r, -0.50 * r, 0.50 * r, r)
    for a in points:
        for b in points:
            # Between crossings of these lines the containment predicate is constant.
            times = {0.0, 1.0}
            for start, end in (
                (a[0], b[0]),
                (a[1], b[1]),
                ((a[0] + a[1]) / math.sqrt(2), (b[0] + b[1]) / math.sqrt(2)),
                ((a[0] - a[1]) / math.sqrt(2), (b[0] - b[1]) / math.sqrt(2)),
            ):
                if abs(end - start) > 1e-12:
                    times.update(
                        t
                        for edge in boundaries
                        if 0 < (t := (edge - start) / (end - start)) < 1
                    )
            ordered = sorted(times)
            probes = [
                ((a[0] * (1 - t) + b[0] * t), (a[1] * (1 - t) + b[1] * t))
                for t in ((x + y) / 2 for x, y in itertools.pairwise(ordered))
            ]
            if not target_contains(probes, side, shape, tolerance):
                return False
    return True


def seat_peg(peg, cube, cube_side):
    """Place the peg coaxially inside the bottom socket of an upright cube."""
    factor = cube_side / REFERENCE_CUBE_SIDE_M
    height = PEG_HEIGHT_M * factor
    x, y, z = cube["default_pos"]
    peg["default_pos"] = [x, y, z - cube_side / 2 + height / 2]
    peg["scale"] = [PEG_DIAMETER_M * factor] * 2 + [height]
    for axis, value in zip(("xlim", "ylim", "zlim"), peg["default_pos"], strict=True):
        peg[axis] = [value, value]


def _basket(layout):
    for group in layout["Rigid"].values():
        for obj in group:
            if obj.get("label") == "basket":
                return obj
    raise ValueError("layout has no basket")


def place_basket_support(
    layout, footprint, depth, *, top=None, clearance=BASKET_POCKET_CLEARANCE_M
):
    """Surround the basket with four static sections leaving a shallow pocket.

    Section tops restore the surrounding tabletop height; their shared inner
    opening is the basket footprint plus one millimetre on each side. Sections
    are created from the target template when the layout has none yet.
    """
    if len(footprint) != 2 or any(not math.isfinite(v) or v <= 0 for v in footprint):
        raise ValueError("basket footprint must be two positive metres")
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("pocket depth must be positive")
    table = layout["Table"]
    if table.get("default_ori", [1, 0, 0, 0]) != [1, 0, 0, 0]:
        raise ValueError("basket support requires an axis-aligned tabletop")
    top = (
        table["default_pos"][2] + table["scale"][2] / 2 + depth if top is None else top
    )
    basket = _basket(layout)
    cx, cy = basket["default_pos"][:2]
    half_x = footprint[0] / 2 + clearance
    half_y = footprint[1] / 2 + clearance
    tx, ty = table["default_pos"][:2]
    sx, sy = table["scale"][:2]
    xmin, xmax, ymin, ymax = tx - sx / 2, tx + sx / 2, ty - sy / 2, ty + sy / 2
    left, right, near, far = cx - half_x, cx + half_x, cy - half_y, cy + half_y
    if not (xmin < left < right < xmax and ymin < near < far < ymax):
        raise ValueError("basket support must lie inside the table")
    sections = layout["Geometry"].get(BASKET_SUPPORT_CATEGORY)
    if not sections:
        template = next(
            obj
            for group in layout["Geometry"].values()
            for obj in group
            if obj.get("label") == "target_zone"
        )
        sections = [deepcopy(template) for _ in range(4)]
        layout["Geometry"][BASKET_SUPPORT_CATEGORY] = sections
    if len(sections) != 4:
        raise ValueError("basket support requires exactly four sections")
    bounds = (
        (xmin, left, ymin, ymax),
        (right, xmax, ymin, ymax),
        (left, right, ymin, near),
        (left, right, far, ymax),
    )
    for section, label, (x0, x1, y0, y1) in zip(
        sections, ("left", "right", "near", "far"), bounds, strict=True
    ):
        position = [(x0 + x1) / 2, (y0 + y1) / 2, top - depth / 2]
        section.update(
            label=f"basket_support_{label}",
            category_idx=0,
            default_pos=position,
            scale=[x1 - x0, y1 - y0, depth],
            physics={
                "type": "geometry",
                "collision": True,
                "static_friction": 0.8,
                "dynamic_friction": 0.8,
                "restitution": 0.0,
            },
            visual={},
            relative_plane="Table",
            margin=0.0,
        )
        for axis, coordinate in zip(("xlim", "ylim", "zlim"), position, strict=True):
            section[axis] = [coordinate, coordinate]
    return sections


def recess_basket(
    layout,
    footprint,
    depth=RECESSED_BASKET_DEPTH_M,
    *,
    clearance=BASKET_POCKET_CLEARANCE_M,
):
    """Seat the basket on a lowered floor and build its surrounding support.

    The tabletop is lowered by ``depth`` and the basket with all of its rigid
    contents follows, preserving their relative poses. The basket is forced
    upright so its footprint matches the pocket. Idempotence is not required;
    callers pass an unmodified reference scene.
    """
    table = layout["Table"]
    if table.get("default_ori", [1, 0, 0, 0]) != [1, 0, 0, 0]:
        raise ValueError("basket support requires an axis-aligned tabletop")
    top = table["default_pos"][2] + table["scale"][2] / 2
    basket = _basket(layout)
    basket_z = top - depth + clearance - BASKET_BOTTOM_LOCAL_Z_M
    shift = basket_z - basket["default_pos"][2]
    for group in layout["Rigid"].values():
        for obj in group:
            obj["default_pos"][2] += shift
    basket.update(
        default_ori=[1.0, 0.0, 0.0, 0.0],
        qpos=[1.0, 0.0, 0.0, 0.0],
        rotate_deg=0.0,
        rotate_rand=False,
    )
    for group in layout["Rigid"].values():
        for obj in group:
            for axis, value in zip(
                ("xlim", "ylim", "zlim"), obj["default_pos"], strict=True
            ):
                obj[axis] = [value, value]
    table["default_pos"][2] -= depth
    layout["Table"].setdefault("task_variation", {})["support"] = {
        "footprint_m": [float(value) for value in footprint],
        "depth_m": float(depth),
        "side_clearance_m": float(clearance),
    }
    place_basket_support(layout, footprint, depth, top=top, clearance=clearance)


def _target_template(layout):
    return next(
        obj
        for group in layout["Geometry"].values()
        for obj in group
        if obj.get("label") == "target_zone"
    )


def add_friction_surface(layout):
    """Rest a static high-friction mat on the tabletop and raise its objects.

    The mat is raised slightly above the table so dynamic objects contact only
    the mat, not the native table surface beneath it. All rigid objects and the
    target marker move up by the mat thickness, preserving their relative
    poses and the task's scoring heights. RoboDojo does not apply per-object
    physics materials on the evaluation path; the mat's USD also carries a
    bound physics material, so the friction is read by PhysX with the collider.
    """
    table = layout["Table"]
    if table.get("default_ori", [1, 0, 0, 0]) != [1, 0, 0, 0]:
        raise ValueError("friction surface requires an axis-aligned tabletop")
    top = table["default_pos"][2] + table["scale"][2] / 2
    for group in layout["Rigid"].values():
        for obj in group:
            obj["default_pos"][2] += SURFACE_THICKNESS_M
            obj["zlim"] = [obj["default_pos"][2]] * 2
    target = _target_template(layout)
    target["default_pos"][2] += SURFACE_THICKNESS_M
    target["zlim"] = [target["default_pos"][2]] * 2
    pad = deepcopy(target)
    position = [*table["default_pos"][:2], top + SURFACE_THICKNESS_M / 2]
    pad.update(
        label="friction_surface",
        category_idx=0,
        default_pos=position,
        scale=[table["scale"][0], table["scale"][1], SURFACE_THICKNESS_M],
        physics={
            "type": "geometry",
            "collision": True,
            "static_friction": SURFACE_STATIC_FRICTION,
            "dynamic_friction": SURFACE_DYNAMIC_FRICTION,
            "restitution": 0.0,
        },
        visual={},
        relative_plane="Table",
        margin=0.0,
    )
    for axis, value in zip(("xlim", "ylim", "zlim"), position, strict=True):
        pad[axis] = [value, value]
    layout["Geometry"][FRICTION_SURFACE_CATEGORY] = [pad]


def place_pinned_fixture(layout, cube, cube_side):
    """Rest the cube on a holed pedestal and pass a free pin through both.

    The pedestal and the cube each have a coaxial vertical hole. The pin runs
    from the tabletop, through the pedestal and the cube, and protrudes above
    the cube so it can be gripped. The pin blocks any horizontal motion of the
    cube, so the cube has to be lifted before it can be transported.
    """
    table = layout["Table"]
    if table.get("default_ori", [1, 0, 0, 0]) != [1, 0, 0, 0]:
        raise ValueError("pinned fixture requires an axis-aligned tabletop")
    if not math.isfinite(cube_side) or cube_side <= 0:
        raise ValueError("pinned fixture requires a positive cube side")
    factor = cube_side / REFERENCE_CUBE_SIDE_M
    height = PEDESTAL_HEIGHT_M * factor
    top = table["default_pos"][2] + table["scale"][2] / 2
    cx, cy = cube["default_pos"][:2]
    layout["Geometry"].pop(PEDESTAL_CATEGORY, None)
    layout["Rigid"].pop(PIN_CATEGORY, None)
    pedestal = deepcopy(_target_template(layout))
    position = [cx, cy, top + height / 2]
    pedestal.update(
        label="pin_pedestal",
        category_idx=0,
        default_pos=position,
        scale=[cube_side, cube_side, height],
        physics={
            "type": "geometry",
            "collision": True,
            "static_friction": 0.8,
            "dynamic_friction": 0.8,
            "restitution": 0.0,
        },
        visual={},
        relative_plane="Table",
        margin=0.0,
    )
    for axis, value in zip(("xlim", "ylim", "zlim"), position, strict=True):
        pedestal[axis] = [value, value]
    layout["Geometry"][PEDESTAL_CATEGORY] = [pedestal]
    pin_height = height + cube_side + PIN_EXPOSED_M * factor
    pin = deepcopy(_target_template(layout))
    position = [cx, cy, top + pin_height / 2]
    pin.update(
        label="retaining_pin",
        category_idx=0,
        default_pos=position,
        default_ori=[1.0, 0.0, 0.0, 0.0],
        qpos=[1.0, 0.0, 0.0, 0.0],
        rotate_deg=0.0,
        rotate_rand=False,
        scale=[PIN_DIAMETER_M * factor] * 2 + [pin_height],
        physics={
            "type": "rigid",
            "mass": 0.05,
            "static_friction": 0.35,
            "dynamic_friction": 0.25,
            "restitution": 0.0,
        },
        visual={},
        relative_plane="Table",
        margin=0.0,
        need_check_stable=False,
    )
    for axis, value in zip(("xlim", "ylim", "zlim"), position, strict=True):
        pin[axis] = [value, value]
    layout["Rigid"][PIN_CATEGORY] = [pin]
    cube["default_pos"][2] = top + height + cube_side / 2


def generate_layout(reference, task, seed):
    """Return a fresh layout; masses, counts, basket scale and controller stay fixed."""
    if task not in TASKS:
        raise ValueError(f"not a project custom task: {task}")
    seed = operator.index(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    layout = deepcopy(reference)
    # Some reference layouts still carry this metadata at the root. RoboDojo
    # treats unknown root keys as object categories; keep it only under Table.
    layout.pop("task_variation", None)
    if task == PINNED_CUBE_TASK:
        # The fixture is rebuilt around the seeded cube below.
        layout["Rigid"].pop(PIN_CATEGORY, None)
        layout["Geometry"].pop(PEDESTAL_CATEGORY, None)
    rigid = [obj for group in layout["Rigid"].values() for obj in group]
    target = next(
        obj
        for group in layout["Geometry"].values()
        for obj in group
        if obj.get("label") == "target_zone"
    )
    table_top = layout["Table"]["default_pos"][2] + layout["Table"]["scale"][2] / 2
    recess_depth = RECESS_DEPTHS.get(
        task, RECESSED_BASKET_DEPTH_M if task in RECESSED_BASKET_TASKS else 0.0
    )
    table_top += recess_depth
    colors = rng.sample(COLORS, len(COLORS))
    shape = rng.choice(SHAPES)
    target_side = (
        rng.uniform(0.37, 0.41) if task in CUBE_TASKS else rng.uniform(0.34, 0.40)
    )
    target_xy = [rng.uniform(0.19, 0.23), rng.uniform(-0.09, -0.01)]
    specs = {}
    if task in BASKET_FAMILY_TASKS:
        basket = next(obj for obj in rigid if obj.get("label") == "basket")
        clearance = (
            BASKET_POCKET_CLEARANCE_M
            if task in RECESSED_BASKET_TASKS
            else BASKET_TABLE_CLEARANCE_M
        )
        basket_z = table_top - recess_depth - BASKET_BOTTOM_LOCAL_Z_M + clearance
        vertical_shift = basket_z - basket["default_pos"][2]
        shift = [rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)]
        for obj in rigid:
            obj["default_pos"][2] += vertical_shift
            obj["default_pos"][:2] = [
                a + b for a, b in zip(obj["default_pos"][:2], shift)
            ]
            if obj is basket:
                obj["default_ori"] = [1.0, 0.0, 0.0, 0.0]
                obj["qpos"] = [1.0, 0.0, 0.0, 0.0]
                obj["rotate_deg"] = 0.0
                obj["rotate_rand"] = False
            if obj is not basket:
                factor = rng.uniform(0.90, 1.0)
                jitter_x = rng.uniform(-0.002, 0.002)
                jitter_y = rng.uniform(-0.002, 0.002)
                # Preserve the reference payload arrangement for the partially
                # loaded basket; random perturbations destabilized its reset.
                if task not in PARTIAL_BASKET_TASKS:
                    obj["scale"] = [s * factor for s in obj["scale"]]
                    obj["default_pos"][0] += jitter_x
                    obj["default_pos"][1] += jitter_y
                if task in HEAVY_BASKET_TASKS:
                    obj["default_pos"][2] = (
                        basket["default_pos"][2] - 0.10 + 0.08 * factor / 2
                    )
            obj.setdefault("visual", {})["color"] = list(
                colors[0 if obj is basket else 1 + rigid.index(obj) % 5]
            )
    else:
        # Independent lanes keep all cube footprints separated, on the reachable tabletop.
        lanes = {
            "large_cube": (-0.22, -0.035),
            "medium_cube": (-0.22, -0.265),
            "small_cube": (0.02, -0.265),
        }
        ranges = {
            "large_cube": (0.19, 0.22),
            "medium_cube": (0.055, 0.070),
            "small_cube": (0.030, 0.040),
        }
        original = {"large_cube": 0.20, "medium_cube": 0.06, "small_cube": 0.035}
        for i, obj in enumerate(rigid):
            label = obj["label"]
            side = rng.uniform(*ranges[label])
            specs[label] = side
            obj["scale"] = [s * side / original[label] for s in obj["scale"]]
            obj["default_pos"] = [
                lanes[label][0] + rng.uniform(-0.015, 0.015),
                lanes[label][1] + rng.uniform(-0.015, 0.015),
                table_top + side / 2,
            ]
            obj.setdefault("visual", {})["color"] = list(colors[i])
    if task in RECESSED_BASKET_TASKS:
        support = reference["Table"]["task_variation"]["support"]
        place_basket_support(
            layout,
            support["footprint_m"],
            recess_depth,
            top=table_top,
            clearance=support["side_clearance_m"],
        )
    elif recess_depth:
        cube = rigid[0]
        cube["default_pos"][2] -= recess_depth
        cx, cy = cube["default_pos"][:2]
        half = specs["large_cube"] / 2 + RECESS_SIDE_CLEARANCE_M
        tx, ty = layout["Table"]["default_pos"][:2]
        sx, sy = layout["Table"]["scale"][:2]
        xmin, xmax, ymin, ymax = tx - sx / 2, tx + sx / 2, ty - sy / 2, ty + sy / 2
        left, right, near, far = cx - half, cx + half, cy - half, cy + half
        bounds = (
            (xmin, left, ymin, ymax),
            (right, xmax, ymin, ymax),
            (left, right, ymin, near),
            (left, right, far, ymax),
        )
        for section, (x0, x1, y0, y1) in zip(
            layout["Geometry"]["recessed_tabletop"], bounds, strict=True
        ):
            section["default_pos"] = [
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                table_top - recess_depth / 2,
            ]
            section["scale"] = [x1 - x0, y1 - y0, recess_depth]
            for axis, value in zip(("xlim", "ylim", "zlim"), section["default_pos"]):
                section[axis] = [value, value]
    if task == PEGGED_CUBE_TASK:
        seat_peg(layout["Geometry"][PEG_CATEGORY][0], rigid[0], specs["large_cube"])
    elif task == PINNED_CUBE_TASK:
        place_pinned_fixture(layout, rigid[0], specs["large_cube"])
    target["default_pos"] = [*target_xy, table_top + 0.00005]
    target["scale"] = [1.0, 1.0, 1.0]  # Generated target mesh is already in metres.
    target["visual"] = {}
    target["physics"] = {"type": "geometry", "collision": False}
    for obj in [*rigid, target]:
        for axis, value in zip(("xlim", "ylim", "zlim"), obj["default_pos"]):
            obj[axis] = [value, value]
    if task in FRICTION_TASKS:
        add_friction_surface(layout)
    layout["Table"]["task_variation"] = {
        "version": 1,
        "seed": seed,
        "task": task,
        "cube_sides": specs,
        "target_side": target_side,
        "target_shape": shape,
        "target_color": list(colors[-1]),
        "pattern": rng.choice(("solid", "checker", "stripes")),
    }
    if task in RECESSED_BASKET_TASKS:
        layout["Table"]["task_variation"]["support"] = deepcopy(
            reference["Table"]["task_variation"]["support"]
        )
    if task in NEW_CUBE_TASKS:
        variation = layout["Table"]["task_variation"]
        variation["fixture"] = deepcopy(reference["Table"]["task_variation"]["fixture"])
        if task in (PEGGED_CUBE_TASK, PINNED_CUBE_TASK):
            factor = specs["large_cube"] / REFERENCE_CUBE_SIDE_M
            variation["fixture"] = {
                key: value * factor for key, value in variation["fixture"].items()
            }
    return layout


def target_mesh(variation):
    """Flat noncolliding USD mesh; no textures, external files, or runtime deps."""
    side = variation["target_side"]
    shape = variation["target_shape"]
    points, faces, colors = [], [], []
    n = 80
    for i in range(n):
        for j in range(n):
            x, y = side * ((i + 0.5) / n - 0.5), side * ((j + 0.5) / n - 0.5)
            if not target_contains([(x, y)], side, shape, tolerance=0):
                continue
            if shape == "circle" and math.hypot(x, y) < side * 0.40:
                continue
            k = len(points)
            h = side / n / 2
            points.extend(
                [
                    (x - h, y - h, 0),
                    (x + h, y - h, 0),
                    (x + h, y + h, 0),
                    (x - h, y + h, 0),
                ]
            )
            faces.extend(range(k, k + 4))
            alternate = (
                (i // 8 + j // 8) % 2
                if variation["pattern"] == "checker"
                else i // 8 % 2
            )
            colors.append(
                tuple(variation["target_color"])
                if variation["pattern"] == "solid" or alternate
                else (0.04, 0.04, 0.04)
            )

    def values(xs):
        return ", ".join(str(x) for x in xs)

    return (
        '#usda 1.0\n( defaultPrim = "Target" )\ndef Xform "Target" {\n def Mesh "Marker" {\n'
        f" point3f[] points = [{values(points)}]\n"
        f" int[] faceVertexCounts = [{values([4] * len(colors))}]\n"
        f" int[] faceVertexIndices = [{values(faces)}]\n"
        f' color3f[] primvars:displayColor = [{values(colors)}] ( interpolation = "uniform" )\n'
        ' uniform token subdivisionScheme = "none"\n bool doubleSided = true\n }\n}\n'
    )
