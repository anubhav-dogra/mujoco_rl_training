# mujoco_rl_training

`mujoco_rl_training` is the non-ROS RL/task layer on top of `mujoco_core`.

It is responsible for:

- environment definitions
- reset logic
- reward logic
- rollout and training executables
- policy save/load artifacts

It is intentionally separate from:

- `mujoco_ros2_driver`
- `mujoco_interface`
- `mujoco_bringup`

because training should consume the simulation backend directly, without ROS in the loop.

## Dependencies

This repo is not a self-contained simulation stack.

It requires the public simulation packages from `mujoco_sim`, in particular:

- `mujoco_core`
- `mujoco_viewer`
- `mujoco_models`

So `mujoco_rl_training` is intended to be built in a workspace together with `mujoco_sim`, not as a fully standalone repo by itself.

## Scope

The current environment is:

- `PendulumEnv`

It provides:

- `reset()`
- `step(double action)`
- `observation() const`

Observation:

- `cos(theta)`
- `sin(theta)`
- `theta_dot`

Action:

- single scalar torque
- clipped to `max_torque`

Reward:

```text
-(theta^2 + 0.1 * theta_dot^2 + 0.001 * torque^2)
```

with `theta` normalized around the upright configuration.

## Design split

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
  - policy training

Do not move reward or episode logic into `mujoco_core`.

## Environment configuration

`PendulumEnvConfig` currently controls:

- `xml_path`
- `simulation_frequency`
- `max_torque`
- `episode_horizon`
- `seed`
- `repeat_action`
- `reset_angle_range`
- `reset_velocity_range`

Important timing detail:

- one env step applies the same action for `repeat_action` MuJoCo steps
- effective env-step duration is:

```text
repeat_action / simulation_frequency
```

For example:

- `repeat_action = 20`
- `simulation_frequency = 1000`

means:

```text
1 env step = 0.02 seconds
```

## Executables

### `pendulum_rl_demo`

Headless random-action smoke test.

Run:

```bash
pixi run pendulum_rl_demo
```

### `pendulum_rl_visual_demo`

Headless environment plus MuJoCo viewer, with random actions.

Run:

```bash
pixi run pendulum_rl_visual_demo
```

### `pendulum_random_search`

Deterministic linear-policy baseline trainer.

It saves:

- `artifacts/pendulum_best_policy.txt`

### `pendulum_reinforce`

First policy-gradient trainer using a Gaussian linear policy.

Run:

```bash
pixi run pendulum_reinforce
```

It saves:

- `artifacts/pendulum_reinforce_policy.txt`

### `pendulum_policy_rollout`

Visual replay for a saved deterministic policy artifact.

Default artifact:

- `artifacts/pendulum_best_policy.txt`

Run default:

```bash
pixi run pendulum_policy_rollout
```

Run a specific artifact:

```bash
pixi run pendulum_policy_rollout artifacts/pendulum_reinforce_policy.txt
```

## Linear-policy random search baseline

This is the first learning baseline. It exists to validate:

- the environment
- the reward
- the rollout loop
- policy save/load workflow
- visual replay

### Policy

The deterministic policy is:

```text
action = w1 * cos(theta) + w2 * sin(theta) + w3 * theta_dot + bias
```

### Search method

The final baseline uses symmetric perturbations:

- start from current best policy
- sample one random perturbation `delta`
- evaluate:
  - `best + delta`
  - `best - delta`
- keep the better one if it beats the current best return

### Evaluation

Each candidate is scored by averaging return over multiple episodes.

That matters because reset is randomized, so one episode is noisy.

### What made the real difference

The main improvements were:

1. Using symmetric perturbations instead of one-sided perturbation.
2. Averaging policy return over multiple episodes instead of trusting one rollout.
3. Saving the best policy and replaying it visually.
4. Matching training and replay environment timing (`repeat_action`, horizon).

This was not about one perfect hyperparameter. It was about reducing noise and making comparison fair.

## REINFORCE implementation

After the random-search baseline, the next phase was a manual REINFORCE implementation.

### Gaussian policy

The stochastic policy is:

```text
mu(s) = w1 * cos(theta) + w2 * sin(theta) + w3 * theta_dot + bias
a ~ Normal(mu(s), sigma^2)
```

Current policy type:

- `PendulumGaussianPolicy`

It provides:

- `mean_action(...)`
- `sample_action(...)`
- `log_probability(...)`

### Trajectory collection

One episode corresponds to one trajectory.

Each step stores:

- observation `s_t`
- sampled action `a_t`
- reward `r_t`
- log-probability `log pi(a_t | s_t)`

Important alignment rule:

- store the observation that produced the action
- not the next observation

### Discounted return

For each time step:

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

Computed backward with:

```text
G_t = r_t + gamma * G_{t+1}
```

### Policy update

The manual gradient is derived for a 1D Gaussian with fixed `sigma`.

Loss intuition:

```text
L = - Σ_t log pi(a_t | s_t) * G_t
```

For the linear Gaussian policy:

```text
∂ log pi(a_t|s_t) / ∂mu_t = (a_t - mu_t) / sigma^2
```

and since:

```text
mu_t = w^T s_t + b
```

the update uses:

```text
coeff_t = (a_t - mu_t) / sigma^2
```

then accumulates:

```text
grad_w += signal_t * coeff_t * s_t
grad_b += signal_t * coeff_t
```

### What initially failed

The naive first REINFORCE version was unstable and performed worse than the zero policy.

That is expected. Vanilla REINFORCE has high variance.

### What made REINFORCE actually start working

These changes mattered:

1. Smaller exploration variance:
   - `sigma = 0.1`
   - not `1.0`

2. Smaller learning rate:
   - `1e-4`
   - not `1e-3`

3. Batch updates:
   - multiple trajectories per update
   - not one episode per update

4. Batch-level baseline:
   - subtract the mean discounted return over the whole batch
   - reduces gradient variance

These changes are what turned REINFORCE from “immediately collapses” into “actually improves”.

### Current training loop

High-level loop:

1. collect multiple trajectories
2. compute discounted returns
3. compute one batch baseline
4. center returns with that baseline
5. accumulate gradients across the batch
6. average the gradient
7. update policy
8. evaluate current mean policy
9. keep best policy seen so far

## Why best-policy tracking exists

There are two different things in the trainers:

### Learning update

This changes the current policy parameters.

Examples:

- random-search replacement
- REINFORCE gradient update

### Best-policy tracking

This is only bookkeeping:

- evaluate current policy
- if it is the best seen so far, save a copy

This is useful because RL updates are noisy. The current policy can get worse after a later update.

## Artifact workflow

This package deliberately avoids hardcoding learned parameters into source files.

Training writes policy artifacts under:

- `artifacts/`

Rollout loads those artifacts.

That keeps training and replay decoupled and avoids manual copy-paste of weights.

## What comes next

The current state is enough to demonstrate:

- environment design
- deterministic baseline learning
- stochastic policy-gradient learning
- policy save/load/replay

The next logical algorithmic step after this REINFORCE version is:

- actor-critic / learned baseline

not more blind hyperparameter tweaking.
