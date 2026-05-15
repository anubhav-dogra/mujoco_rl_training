# Torch Examples

These examples are a small learning path for using LibTorch from C++ 

For a function-by-function explanation of the Torch APIs used here, see `TORCH_API_NOTES.md`.

Reusable Torch/RL pieces live under:

```text
include/mujoco_rl_training/torch/
```

## 1. Tensor Basics

File: `torch_tensor_basics.cpp`

This example introduces tensors, tensor shapes, devices, indexing, reshaping, and broadcasting.

Key ideas:

- A tensor is the basic numerical object in Torch.
- `.to(device)` moves a tensor to CPU or CUDA.
- `sizes()` shows the tensor shape.
- `reshape({rows, cols})` changes the shape view.
- `index(...)` extracts rows, columns, or slices.
- Adding a vector to a matrix can broadcast the vector across rows.

RL connection:

- Observations become tensors such as `[batch_size, obs_dim]`.
- Actions become tensors such as `[batch_size, action_dim]`.
- Rewards and returns usually become `[batch_size, 1]`.

## 2. Autograd Basics

File: `torch_autograd_basics.cpp`

This example introduces automatic differentiation.

It computes:

```text
loss = w^2 + 3w
```

For `w = 2`, the derivative is:

```text
d(loss)/dw = 2w + 3 = 7
```

Key ideas:

- `requires_grad()` tells Torch to track operations on a tensor.
- `loss.backward()` computes gradients through the computation graph.
- `w.grad()` stores `d(loss)/dw`.
- Only floating point or complex tensors can require gradients.

RL connection:

- Neural actor and critic parameters need gradients.
- Policy loss and value loss are scalar losses.
- `backward()` gives parameter gradients for the optimizer.

## 3. Manual Gradient Descent

File: `torch_manual_grad_descent.cpp`

This example trains one scalar parameter manually.

Goal:

```text
2w = 10
```

So the correct value is:

```text
w = 5
```

Training loop:

```text
forward -> loss -> backward -> manual update -> zero gradient
```

Key ideas:

- `loss.backward()` accumulates gradients.
- `torch::NoGradGuard` prevents the parameter update itself from being recorded in autograd.
- `w.grad().zero_()` clears old gradients before the next step.
- Without clearing gradients, Torch keeps accumulating them.

RL connection:

- This is the same update idea as policy gradient, but with one scalar parameter.
- Later, optimizers will update thousands of actor/critic parameters instead of one scalar.

## 4. Linear Regression With Optimizer

File: `torch_linear_regression_with_opt.cpp`

This example trains a linear model:

```text
y = wx + b
```

on fake data:

```text
y = 2x + 1
```

Key ideas:

- `torch::nn::Linear(1, 1)` creates trainable `weight` and `bias`.
- `model->parameters()` gives all trainable tensors to the optimizer.
- `optimizer.zero_grad()` clears gradients.
- `optimizer.step()` updates parameters.
- `torch::mse_loss(prediction, target)` measures supervised regression error.

RL connection:

- A critic is also a regression model.
- Instead of predicting `y = 2x + 1`, the critic predicts return/value.
- Optimizers replace our manual parameter update code.

## 5. MLP Critic

File: `torch_mlp_critic.cpp`

This example trains a small neural network critic:

```text
observation -> predicted value
```

Current fake task:

```text
input:  obs with shape [256, 6]
target: returns = sum(obs), shape [256, 1]
```

Model:

```text
6 inputs -> Linear(6, 64) -> tanh -> Linear(64, 64) -> tanh -> Linear(64, 1)
```

Key ideas:

- `CriticImpl` defines the module implementation.
- `TORCH_MODULE(Critic)` creates the convenient shared-pointer wrapper.
- `register_module(...)` makes submodules visible to Torch.
- The output is one scalar per observation.
- The critic is trained with MSE between predicted values and target returns.

RL connection:

- In VPG/actor-critic, the critic estimates `V(s)`.
- `V(s)` is the expected future return starting from state `s`.
- The actor uses the critic to compute advantage:

```text
advantage = return - V(s)
```

This reduces policy-gradient variance compared with using raw returns directly.

## Important Caveat

The MLP critic example trains on random normal observations:

```text
obs ~ N(0, 1)
```

If you test it on `[1, 2, 3, 4, 5, 6]`, that input is outside the training distribution, so prediction can be poor even if training loss is low. Test with another `torch::randn({1, 6})` sample first.

## Reusable Headers

`mujoco_rl_training/torch/Actor.h` provides `TorchActor`, an MLP that maps observations to mean actions.

`mujoco_rl_training/torch/Critic.h` provides `TorchCritic`, an MLP that maps observations to scalar value estimates.

`mujoco_rl_training/torch/GaussianPolicy.h` provides Gaussian policy helpers:

- `sample_gaussian_action(mean, log_std)`
- `gaussian_log_prob(action, mean, log_std)`
- `squash_action(raw_action)`

`mujoco_rl_training/torch/TensorUtils.h` provides `default_device()`, which chooses CUDA when available and otherwise falls back to CPU.
