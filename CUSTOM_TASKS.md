# Custom RoboDojo tasks

This repository branch contains the project-specific task definitions and runtime support used alongside the standard RoboDojo tasks. The standard task definitions remain registered and unchanged.

## Task catalog

### Basket tasks

- `basket_transport_empty`
- `basket_transport_partial`
- `basket_transport_overloaded`
- `basket_transport_heavy_cubes`
- `basket_transport_fixed_cube`
- `recessed_basket`
- `recessed_heavy_basket`
- `basket_transport_empty_high_friction`
- `basket_transport_partial_high_friction`
- `basket_transport_heavy_cubes_high_friction`

### Tool tasks

- `tool_push_cube`
- `tool_pull_cube`

Both use one rigid `l_tool` asset: a 0.40 m long arm with a crossbar at the far
end and a raised handle at the near end. In `tool_push_cube` the cube starts
near the robot and the target is beyond the arms' reach; the agent pushes the
cube with the crossbar. In `tool_pull_cube` the cube starts beyond reach and the
target is near; the long arm passes beside the cube so the crossbar hooks its
far face and the agent pulls it back. The cube must never be lifted above 2 cm.
The checked-in asset is `task/RoboDojo/custom_assets/l_tool/00000/`; regenerate
it with `python scripts/custom_tasks/l_tool_asset.py --output
task/RoboDojo/custom_assets/l_tool/00000`.

### Cube tasks

- `cube_transport_large`
- `cube_stack_sizes`
- `cube_transport_recessed`
- `cube_transport_recessed_deep`
- `cube_transport_recessed_fragile`
- `cube_transport_pegged`
- `cube_transport_pinned`
- `cube_transport_high_friction`

The matching task YAML files are in `task/RoboDojo/config/`; task implementations and shared helpers are in `task/RoboDojo/tasks/`. The `dual_x5_payload_limited` robot configuration is registered in `env_cfg/robot/`.

## Seeded scenes and assets

Task definitions use RoboDojo's normal task and layout loading paths. Seed-specific layouts and generated fixture or target assets are materialized into `Assets/` and `Assets/Eval_Layout/` by the companion project setup scripts. The checked-in `custom_variant_reference_v1/` files preserve the reference layouts those generators start from. Generated seed outputs are not source files and are not committed here. The L-tool asset is source and is checked in under `task/RoboDojo/custom_assets/`; the seed installer copies it into `Assets/Object/RoboDojo/Rigid/l_tool/00000/`.

Use the companion project's `install_seeded_variants.py` and task-specific asset setup scripts to prepare the requested tasks and seeds in a RoboDojo checkout. Keep the generated layout and asset files together with that checkout when running evaluation.

## Runtime integration

The custom tasks rely on the registered limited-payload robot configuration, task-scoped rigid-body properties, seeded layout selection, and the RoboDojo evaluation client's runtime action and benchmark evaluation interfaces. The high-friction reset stability check is included in the layout manager.

## Task catalog and evaluation

`task/RoboDojo/task_catalog.json` is the task metadata source. Each custom
entry records physical principles, acquisition status, evaluator version and
proposed physical controls. Update an acquisition status only against
evidence for the deployed task revision, seed and controller.

Custom evaluator version `custom-progress-v2` awards 100 for full native
success and uses the highest satisfied partial tier for unsuccessful episodes.
The first two tiers require the closest planar target distance to drop by 25%
or 50% of the settled start distance, with at least 1 cm of progress. Later
tiers depend on terminal placement, orientation, payload containment or
stack levels, depending on the task. A transient transport milestone can earn
credit even if the object subsequently moves away from the goal. Terminal
milestones still require the physical arrangement at evaluation.

RoboDojo samples the transported object's pose after every controlled
physics step. The no-lift rule for L-tool tasks compares peak cube center
height with the settled start, allowing a rise of at most 2 cm. The fragile
cube allows at most 10 degrees of top-face tilt throughout the trajectory.
Either violation prevents full or partial credit for the corresponding task.
Tool and pin motion summaries are available as evidence of object motion.
They do not determine which contact moved the cube.

Each custom result records the milestone labels, cumulative points and which
tiers received credit, alongside the official score and success flag.
