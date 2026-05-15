# Torch API Notes

This file explains the LibTorch functions and classes used in the examples.

The examples are intentionally small, so the goal here is not to cover all of Torch. The goal is to understand the exact API surface needed for actor-critic RL.

## Tensor Basics

### `torch::Tensor`

`torch::Tensor` is the main numerical array type in Torch.

It can represent:

- a scalar: shape `[]`
- a vector: shape `[N]`
- a matrix: shape `[N, M]`
- a batch of observations: shape `[batch_size, obs_dim]`

Example:

```cpp
torch::Tensor x = torch::tensor({1.0, 2.0, 3.0});
```

RL mapping:

```text
obs      -> Tensor [batch_size, obs_dim]
actions  -> Tensor [batch_size, action_dim]
returns  -> Tensor [batch_size, 1]
values   -> Tensor [batch_size, 1]
```

### `torch::tensor(...)`

Creates a tensor from explicit values.

```cpp
auto x = torch::tensor({1.0, 2.0, 3.0});
```

Use floating point values like `1.0`, not `1`, when gradients are needed.

Wrong for autograd:

```cpp
torch::tensor({2}, torch::requires_grad());
```

Correct:

```cpp
torch::tensor({2.0}, torch::requires_grad());
```

### `torch::eye(n)`

Creates an identity matrix.

```cpp
auto identity = torch::eye(3);
```

Output shape:

```text
[3, 3]
```

### `torch::arange(start, end)`

Creates evenly spaced values from `start` to `end - 1`.

```cpp
auto m = torch::arange(0, 6);
```

Output:

```text
[0, 1, 2, 3, 4, 5]
```

Often followed by `reshape`.

### `torch::linspace(start, end, steps)`

Creates `steps` evenly spaced values including both endpoints.

```cpp
auto x = torch::linspace(-1.0, 1.0, 100);
```

Used in the linear regression example to make training data.

### `torch::randn(shape)`

Creates random samples from a normal distribution.

```cpp
auto obs = torch::randn({256, 6});
```

Meaning:

```text
256 observations
6 values per observation
sampled from N(0, 1)
```

RL mapping:

```text
fake batch of observations
```

### `torch::randn_like(tensor)`

Creates random normal values with the same shape as another tensor.

```cpp
auto noise = torch::randn_like(mean_action);
```

If `mean_action` is `[batch_size, action_dim]`, then `noise` has the same shape.

Used for Gaussian policy sampling:

```cpp
action = mean + std * noise;
```

### `torch::full(shape, value)`

Creates a tensor filled with a constant value.

```cpp
auto log_std = torch::full({action_dim}, -0.5);
```

This creates one log standard deviation per action dimension.

For `action_dim = 2`:

```text
log_std shape = [2]
```

Then:

```cpp
auto std = torch::exp(log_std);
```

converts log standard deviation into standard deviation.

## Shape Operations

### `.sizes()`

Returns the tensor shape.

```cpp
std::cout << x.sizes() << '\n';
```

Shape debugging is critical in Torch. Many errors are shape errors.

Example error:

```text
mat1 and mat2 shapes cannot be multiplied (4x64 and 6x64)
```

This means a linear layer expected `6` input features but received `64`.

### `.reshape({rows, cols})`

Changes tensor shape.

```cpp
auto x = torch::linspace(-1.0, 1.0, 100).reshape({100, 1});
```

This converts:

```text
[100]
```

into:

```text
[100, 1]
```

That matters because `torch::nn::Linear(1, 1)` expects input shaped:

```text
[batch_size, input_dim]
```

### `.index(...)`

Extracts rows, columns, or slices.

```cpp
m.index({0});
m.index({torch::indexing::Slice(), 1});
```

Meaning:

```text
row 0
all rows, column 1
```

### `torch::indexing::Slice()`

Represents `:` slicing, similar to Python/NumPy.

```cpp
m.index({torch::indexing::Slice(), 1});
```

Equivalent idea:

```text
m[:, 1]
```

### `.sum(dim, keepdim)`

Sums values along a dimension.

```cpp
auto returns = obs.sum(1, true);
```

If `obs` shape is:

```text
[256, 6]
```

then `obs.sum(1, true)` is:

```text
[256, 1]
```

because dimension `1` is the feature dimension.

`keepdim = true` keeps the output compatible with critic output shape `[batch_size, 1]`.

### `.mean()`

Computes the mean of all values.

```cpp
auto actor_loss = -(log_prob * advantages).mean();
```

This converts a batch of per-sample losses into one scalar loss for `backward()`.

## Device Operations

The shared helper is defined in:

```text
include/mujoco_rl_training/torch/TensorUtils.h
```

Use it when examples or training programs should automatically prefer CUDA:

```cpp
const auto device = mujoco_rl_training::default_device();
```

### `torch::cuda::is_available()`

Checks if CUDA is usable from Torch.

```cpp
auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
```

Use this before moving tensors/models to GPU.

### `torch::kCPU` and `torch::kCUDA`

Device constants.

```cpp
torch::Device(torch::kCUDA);
torch::Device(torch::kCPU);
```

### `.to(device)`

Moves a tensor or module to CPU/GPU.

```cpp
auto x = torch::tensor({1.0, 2.0, 3.0}).to(device);
model->to(device);
```

Important rule:

```text
all tensors in one operation must be on the same device
```

Do not mix CPU observations with CUDA model weights.

### `.cpu()`

Moves tensor to CPU.

```cpp
std::cout << tensor.cpu() << '\n';
```

Useful before printing or converting data back to standard C++ types.

### `.item<double>()`

Extracts a scalar tensor into a C++ value.

```cpp
double loss_value = loss.item<double>();
```

Only use this for scalar tensors.

## Math Operations

### `torch::pow(x, exponent)`

Raises values to a power.

```cpp
auto loss = torch::pow(prediction - 10.0, 2);
```

Used for squared error in manual gradient descent.

### `torch::exp(x)`

Computes elementwise exponential.

```cpp
auto std = torch::exp(log_std);
```

This is common in Gaussian policies because standard deviation must be positive.

Instead of learning `std` directly, policies usually learn or store:

```text
log_std
```

Then:

```text
std = exp(log_std)
```

### `torch::tanh(x)`

Applies hyperbolic tangent elementwise.

In networks:

```cpp
x = torch::tanh(fc1->forward(x));
```

This gives nonlinearity.

For actions:

```cpp
normalized_action = torch::tanh(raw_action);
```

This squashes actions into:

```text
[-1, 1]
```

Then actions can be scaled by torque limits.

### `torch::mse_loss(prediction, target)`

Mean squared error loss.

```cpp
auto loss = torch::mse_loss(values, returns);
```

Used for critic training:

```text
critic should predict return
```

## Autograd

### `torch::requires_grad()`

Marks a tensor as trainable.

```cpp
auto w = torch::tensor({2.0}, torch::requires_grad());
```

Torch will track operations involving `w`.

### `.backward()`

Computes gradients from a scalar loss.

```cpp
loss.backward();
```

After this, trainable tensors/modules have gradients.

### `.grad()`

Accesses a tensor's gradient.

```cpp
w.grad();
```

For module parameters, optimizers read gradients internally.

### `.zero_()`

In-place fill with zero.

```cpp
w.grad().zero_();
```

Gradients accumulate by default, so they must be cleared between training steps.

The trailing underscore means in-place operation.

### `torch::NoGradGuard`

Temporarily disables autograd recording.

```cpp
{
    torch::NoGradGuard no_grad;
    w -= 0.01 * w.grad();
}
```

Manual parameter updates should not become part of the computation graph.

### `.detach()`

Returns a tensor disconnected from the computation graph.

```cpp
auto mean_before = actor->forward(obs).detach();
```

Useful for logging values before/after an update without tracking gradients.

It is also important in score-function policy gradients:

```cpp
auto action = sample_gaussian_action(mean, log_std).detach();
auto log_prob = gaussian_log_prob(action, mean, log_std);
```

Here the sampled action is treated as fixed while updating the probability of that action. If the sampled action keeps its gradient path through `mean`, then `(action - mean)` can cancel and the example may produce no useful actor update.

## Neural Network Modules

The reusable actor and critic modules are defined in:

```text
include/mujoco_rl_training/torch/Actor.h
include/mujoco_rl_training/torch/Critic.h
```

### `torch::nn::Module`

Base class for custom neural networks.

```cpp
struct CriticImpl : torch::nn::Module {
    ...
};
```

### `torch::nn::Linear(in, out)`

Fully connected layer:

```text
y = xW^T + b
```

Example:

```cpp
torch::nn::Linear fc1{nullptr};
fc1 = register_module("fc1", torch::nn::Linear(6, 64));
```

Shape rule:

```text
input  [batch_size, in]
output [batch_size, out]
```

### `register_module(name, module)`

Registers a submodule so Torch can find its parameters.

```cpp
fc1 = register_module("fc1", torch::nn::Linear(obs_dim, 64));
```

Without registration, `parameters()` will not include that layer.

### `TORCH_MODULE(Name)`

Creates the convenient module-holder type.

```cpp
struct CriticImpl : torch::nn::Module {
    ...
};

TORCH_MODULE(Critic);
```

Then use:

```cpp
Critic critic(6);
critic->forward(obs);
critic->parameters();
```

This matches built-in LibTorch style:

```cpp
torch::nn::Linear linear(6, 64);
linear->forward(x);
```

### `forward(...)`

Defines the computation performed by the module.

```cpp
torch::Tensor forward(torch::Tensor obs) {
    auto x = torch::tanh(fc1->forward(obs));
    x = torch::tanh(fc2->forward(x));
    return mean->forward(x);
}
```

For an actor:

```text
obs -> mean action
```

For a critic:

```text
obs -> value estimate
```

### `parameters()`

Returns trainable parameters from a module.

```cpp
torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(0.01));
```

## Optimizers

### `torch::optim::Adam`

Adaptive optimizer commonly used for neural networks.

```cpp
torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.01));
```

### `optimizer.zero_grad()`

Clears old gradients before the next update.

```cpp
optimizer.zero_grad();
```

### `optimizer.step()`

Updates parameters using current gradients.

```cpp
optimizer.step();
```

Training loop pattern:

```text
prediction -> loss -> zero_grad -> backward -> step
```

## Gaussian Policy Functions

The reusable Gaussian helpers are defined in:

```text
include/mujoco_rl_training/torch/GaussianPolicy.h
```

### `log_std`

`log_std` stores the logarithm of action standard deviation.

```cpp
auto log_std = torch::full({action_dim}, -0.5);
auto std = torch::exp(log_std);
```

This is numerically convenient because:

```text
std must be positive
exp(any real number) is positive
```

### Gaussian sampling

```cpp
auto action = mean + std * torch::randn_like(mean);
```

This samples:

```text
action ~ Normal(mean, std)
```

Shape example:

```text
mean   [batch_size, action_dim]
std    [action_dim]
action [batch_size, action_dim]
```

`std` broadcasts across the batch.

For policy-gradient log-probability training, detach the sampled action before recomputing `log_prob`:

```cpp
auto action = sample_gaussian_action(mean, log_std).detach();
```

### Gaussian log probability

Formula:

```text
log_prob = -0.5 * (((action - mean) / std)^2 + 2*log_std + log(2*pi))
```

Code:

```cpp
auto log_prob_per_dim =
    -0.5 * (torch::pow((action - mean) / std, 2) + 2.0 * log_std + kLogTwoPi);
```

For multi-dimensional actions:

```cpp
return log_prob_per_dim.sum(1, true);
```

This sums log probabilities across action dimensions.

Output shape:

```text
[batch_size, 1]
```

### Policy-gradient actor loss

```cpp
auto actor_loss = -(log_prob * advantages).mean();
```

Meaning:

```text
positive advantage -> increase probability of action
negative advantage -> decrease probability of action
```

This is the core VPG actor update.

## Shape Checklist For RL

Use this checklist when debugging actor-critic code:

```text
obs          [batch_size, obs_dim]
mean         [batch_size, action_dim]
std          [action_dim] or [batch_size, action_dim]
action       [batch_size, action_dim]
log_prob     [batch_size, 1]
advantages   [batch_size, 1]
actor_loss   scalar
values       [batch_size, 1]
returns      [batch_size, 1]
critic_loss  scalar
```

If a matrix multiplication error appears, check the last dimension of the input tensor against the input dimension of the `Linear` layer.
