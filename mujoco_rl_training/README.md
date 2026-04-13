# mujoco_rl_training

`mujoco_rl_training` is the non-ROS RL/task layer on top of `mujoco_core`.

It is responsible for:

- environment definitions
- reset logic
- reward logic
- rollout/demo executables

It is intentionally separate from:

- `mujoco_ros2_driver`
- `mujoco_interface`
- `mujoco_bringup`

because RL should consume the simulation backend directly without ROS in the loop.

## Current Scope

The first environment is:

- `PendulumEnv`

Current public API:

- `reset()`
- `step(double action)`
- `observation() const`

Current observation:

- `cos(theta)`
- `sin(theta)`
- `theta_dot`

Current action:

- one scalar torque command
- clipped to `max_torque`

Current reward:

```text
-(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
```

with `theta` normalized around upright.

## Design Split

Keep this split strict:

- `mujoco_core`
  - plant backend
  - MuJoCo model/data ownership
  - stepping
  - command application
  - raw state access

- `mujoco_rl_training`
  - task semantics
  - reward
  - reset randomization
  - rollout utilities

Do not move reward or episode logic into `mujoco_core`.

## Demo

The package currently provides:

- `pendulum_rl_demo`
- `pendulum_rl_visual_demo`

Run it with:

```bash
pixi run pendulum_rl_demo
```

That executable:

- resolves the installed pendulum MuJoCo model
- constructs `PendulumEnv`
- resets once
- applies random torques for a short rollout
- prints observation and reward per step

Run the visual rollout with:

```bash
pixi run pendulum_rl_visual_demo
```

That executable:

- reuses `PendulumEnv`
- opens a MuJoCo viewer window
- applies random torques while rendering the shared simulation state
- supports viewer pause/reset controls

## Expected Next Steps

Likely next improvements:

- tighten `reset()` / `step()` state access into fewer lock scopes
- add a simple policy rollout executable
- add tests for environment reset and reward behavior
- add more tasks/environments once the pendulum API is stable
