"""Author the project's long L-shaped push/pull tool asset.

The tool is a flat long arm with a crossbar at its far end and a raised handle
at the near end.  It is authored in a unit cube so a replay record can set the
physical dimensions through ``scale``:

    x = 0.13 m (tool width), y = 0.40 m (tool length), z = 0.12 m (tool height)

The long arm is offset to one side of the crossbar.  Consequently the arm can
pass beside the cube while the crossbar reaches behind it, which is what makes
the same asset usable for both pushing and pulling.  The handle is tall enough
for a top-down grasp; the flat arm and crossbar stay on the tabletop.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path

TOOL_CATEGORY = "l_tool"
TOOL_LENGTH_M = 0.40
TOOL_WIDTH_M = 0.13
TOOL_HEIGHT_M = 0.12
TOOL_MASS_KG = 0.35
TOOL_STATIC_FRICTION = 0.5
TOOL_DYNAMIC_FRICTION = 0.4

# Normalised layout in the unit cube.  The crossbar defines the far end and
# both x extents; the handle defines the top of the bounding box.
ARM_X = (0.26, 0.46)
CROSSBAR_X = (-0.5, 0.5)
CROSSBAR_Y = (0.425, 0.5)
HANDLE_X = (0.245, 0.475)
HANDLE_Y = (-0.5, -0.4)
BODY_Z = (-0.5, -1.0 / 3.0)
HANDLE_Z = (-1.0 / 3.0, 0.5)

# The crossbar's inner face is this far in front of the asset origin.  The
# replay records use it to leave a small contact gap to the cube.
CROSSBAR_INSET_M = CROSSBAR_Y[0] * TOOL_LENGTH_M
# The long arm's centre is this far to one side of the asset origin.
ARM_CENTER_OFFSET_M = (ARM_X[0] + ARM_X[1]) / 2 * TOOL_WIDTH_M
ARM_HALF_WIDTH_M = (ARM_X[1] - ARM_X[0]) / 2 * TOOL_WIDTH_M
HANDLE_CENTER_Y_M = (HANDLE_Y[0] + HANDLE_Y[1]) / 2 * TOOL_LENGTH_M


def _box(name, x0, x1, y0, y1, z0, z1, color):
    points = list(product((x0, x1), (y0, y1), (z0, z1)))
    # point order: (-x,-y,-z), (-x,-y,+z), (-x,+y,-z), (-x,+y,+z),
    #              (+x,-y,-z), (+x,-y,+z), (+x,+y,-z), (+x,+y,+z)
    faces = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    return (
        f'    def Mesh "{name}" (\n'
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n'
        "    ) {\n"
        '        uniform token physics:approximation = "convexHull"\n'
        f"        point3f[] points = {[tuple(round(value, 6) for value in p) for p in points]}\n"
        f"        int[] faceVertexCounts = {[len(face) for face in faces]}\n"
        f"        int[] faceVertexIndices = {[index for face in faces for index in face]}\n"
        f"        color3f[] primvars:displayColor = [{tuple(color)}]\n"
        '        uniform token subdivisionScheme = "none"\n'
        "        rel material:binding = </LTool/Looks/ToolMaterial>\n"
        "    }\n"
    )


def l_tool_usd() -> str:
    arm = _box("LongArm", *ARM_X, -0.5, 0.5, *BODY_Z, (0.24, 0.26, 0.29))
    crossbar = _box("Crossbar", *CROSSBAR_X, *CROSSBAR_Y, *BODY_Z, (0.24, 0.26, 0.29))
    handle = _box("Handle", *HANDLE_X, *HANDLE_Y, *HANDLE_Z, (0.95, 0.45, 0.10))
    looks = (
        '    def Scope "Looks" {\n'
        '        def Material "ToolMaterial" {\n'
        "            token outputs:surface.connect = "
        "</LTool/Looks/ToolMaterial/PreviewSurface.outputs:surface>\n"
        '            def Shader "PrimvarReader" {\n'
        '                uniform token info:id = "UsdPrimvarReader_float3"\n'
        '                token inputs:varname = "displayColor"\n'
        "                float3 outputs:result\n"
        "            }\n"
        '            def Shader "PreviewSurface" {\n'
        '                uniform token info:id = "UsdPreviewSurface"\n'
        "                color3f inputs:diffuseColor.connect = "
        "</LTool/Looks/ToolMaterial/PrimvarReader.outputs:result>\n"
        "                float inputs:roughness = 0.55\n"
        "                float inputs:metallic = 0.1\n"
        "                token outputs:surface\n"
        "            }\n"
        "        }\n"
        "    }\n"
    )
    return (
        '#usda 1.0\n( defaultPrim = "LTool" )\n'
        'def Xform "LTool" (prepend apiSchemas = ["PhysxContactReportAPI"]) {\n'
        "    float physxContactReport:threshold = 0\n"
        f"{arm}{crossbar}{handle}{looks}}}\n"
    )


def l_tool_metadata() -> dict:
    bounds = {
        "extents": [1.0, 1.0, 1.0],
        "vertices": [list(point) for point in product((-0.5, 0.5), repeat=3)],
    }
    return {
        "physics": {
            "mass": TOOL_MASS_KG,
            "friction": TOOL_STATIC_FRICTION,
            "size": [1.0, 1.0, 1.0],
        },
        "geometry": {
            "aligned_bbox": bounds,
            "oriented_bbox": bounds,
            "bbox": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "radius": math.sqrt(3.0) / 2.0,
        },
        "active": {},
    }


def write_asset(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    usd = directory / "object.usd"
    metadata = directory / "metadata.json"
    usd.write_text(l_tool_usd())
    metadata.write_text(json.dumps(l_tool_metadata(), indent=2) + "\n")
    return [usd, metadata]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="asset directory, e.g. task/RoboDojo/custom_assets/l_tool/00000",
    )
    args = parser.parse_args()
    for path in write_asset(args.output):
        print(path)


if __name__ == "__main__":
    main()
