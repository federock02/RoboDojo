"""Versioned, CPU-only seeded layouts for the seven project-owned tasks."""

import itertools
import math
import operator
import random
from copy import deepcopy

BASKET_TASKS = tuple(
    "basket_transport_" + s
    for s in ("empty", "partial", "overloaded", "heavy_cubes", "fixed_cube")
)
CUBE_TASKS = ("cube_transport_large", "cube_stack_sizes")
TASKS = BASKET_TASKS + CUBE_TASKS
COLORS = (
    (0.95, 0.22, 0.05),
    (0.05, 0.4, 0.9),
    (0.15, 0.75, 0.25),
    (0.8, 0.15, 0.65),
    (0.95, 0.75, 0.05),
    (0.05, 0.75, 0.75),
)
SHAPES = ("square", "round", "cross", "circle")


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


def generate_layout(reference, task, seed):
    """Return a fresh layout; masses, counts, basket scale and controller stay fixed."""
    if task not in TASKS:
        raise ValueError(f"not a project custom task: {task}")
    seed = operator.index(seed)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    layout = deepcopy(reference)
    rigid = [obj for group in layout["Rigid"].values() for obj in group]
    target = next(
        obj
        for group in layout["Geometry"].values()
        for obj in group
        if obj.get("label") == "target_zone"
    )
    table_top = layout["Table"]["default_pos"][2] + layout["Table"]["scale"][2] / 2
    colors = rng.sample(COLORS, len(COLORS))
    shape = rng.choice(SHAPES)
    target_side = (
        rng.uniform(0.37, 0.41) if task in CUBE_TASKS else rng.uniform(0.34, 0.40)
    )
    target_xy = [rng.uniform(0.19, 0.23), rng.uniform(-0.09, -0.01)]
    specs = {}
    if task in BASKET_TASKS:
        basket = next(obj for obj in rigid if obj.get("label") == "basket")
        shift = [rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)]
        for obj in rigid:
            obj["default_pos"][:2] = [
                a + b for a, b in zip(obj["default_pos"][:2], shift)
            ]
            if obj is not basket:
                # Shrink only: retain the validated floor spacing and handle clearance.
                factor = rng.uniform(0.90, 1.0)
                obj["scale"] = [s * factor for s in obj["scale"]]
                obj["default_pos"][0] += rng.uniform(-0.002, 0.002)
                obj["default_pos"][1] += rng.uniform(-0.002, 0.002)
                if task.endswith(("heavy_cubes", "fixed_cube")):
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
    target["default_pos"] = [*target_xy, table_top + 0.00005]
    target["scale"] = [1.0, 1.0, 1.0]  # Generated target mesh is already in metres.
    target["visual"] = {}
    target["physics"] = {"type": "geometry", "collision": False}
    for obj in [*rigid, target]:
        for axis, value in zip(("xlim", "ylim", "zlim"), obj["default_pos"]):
            obj[axis] = [value, value]
    layout["task_variation"] = {
        "version": 1,
        "seed": seed,
        "task": task,
        "cube_sides": specs,
        "target_side": target_side,
        "target_shape": shape,
        "target_color": list(colors[-1]),
        "pattern": rng.choice(("solid", "checker", "stripes")),
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
        '#usda 1.0\n( defaultPrim = "Target" )\ndef Mesh "Target" {\n'
        f" point3f[] points = [{values(points)}]\n"
        f" int[] faceVertexCounts = [{values([4] * len(colors))}]\n"
        f" int[] faceVertexIndices = [{values(faces)}]\n"
        f' color3f[] primvars:displayColor = [{values(colors)}] ( interpolation = "uniform" )\n'
        ' uniform token subdivisionScheme = "none"\n bool doubleSided = true\n}\n'
    )
