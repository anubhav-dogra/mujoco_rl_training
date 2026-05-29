#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

// =============================================================================
// Squashed Gaussian policy utilities.
//
// PPO's continuous-action policy here is a Gaussian whose sample is pushed
// through tanh before being scaled to the env's action range. The sequence:
//
//   raw_action ~ Normal(mean, exp(log_std))     [pre-squash; unbounded]
//   squashed   = tanh(raw_action)               [in (-1, 1)]
//   command    = scale_action(squashed, ...)    [in env's actual range]
//
// The policy's *probability density* of taking action `a` is the density of
// the corresponding raw_action under the Gaussian, MINUS a correction for the
// change of variables introduced by tanh (because pdf changes through
// nonlinear transforms by the Jacobian determinant).
//
// We store and pass `raw_action` (the pre-tanh sample) around inside PPO,
// because the log-prob formula for the squashed distribution is cleanest in
// terms of the raw action.
// =============================================================================

// Diagonal Gaussian log-probability of `action` under N(mean, exp(log_std)).
//
// For a 1-D Gaussian:
//   log N(x | mu, sigma) = -0.5 * ((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
//
// Here `2.0 * log_std` is `2 * log(sigma) = log(sigma^2)`, but written that
// way to avoid recomputing log of `std`.
//
// For diagonal multivariate Gaussian, sum log-probs across action dimensions.
// Inputs have shape [batch, action_dim]; returns [batch, 1] (sum kept as a
// column so it lines up with PPO's stored old_log_probs column tensor).
inline torch::Tensor gaussian_log_prob(const torch::Tensor& action, const torch::Tensor& mean,
                                       const torch::Tensor& log_std) {
    constexpr double kLogTwoPi = 1.8378770664093453;  // log(2*pi)
    const auto std = torch::exp(log_std);
    const auto log_prob_per_dim =
        -0.5 * (torch::pow((action - mean) / std, 2) + 2.0 * log_std + kLogTwoPi);
    return log_prob_per_dim.sum(1, true);  // sum over action dim, keep dim
}

// Reparameterization-style sample from N(mean, exp(log_std)).
//   x = mu + sigma * eps,  eps ~ N(0, I)
// This form keeps gradients flowing through `mean`/`log_std` when needed
// (PPO doesn't backprop through the sample, but the form is the same).
inline torch::Tensor sample_gaussian_action(const torch::Tensor& mean, const torch::Tensor& log_std) {
    return mean + torch::exp(log_std) * torch::randn_like(mean);
}

// tanh squashing maps the unbounded raw_action to (-1, 1).
inline torch::Tensor squash_action(const torch::Tensor& raw_action) {
    return torch::tanh(raw_action);
}

// Log-probability of the squashed action y = tanh(x), given x = raw_action.
//
// Change-of-variables for monotonic transforms:
//   p_y(y) = p_x(x) / |dy/dx|
//   log p_y(y) = log p_x(x) - log(|dy/dx|)
//
// For y = tanh(x):  dy/dx = 1 - tanh(x)^2 = 1 - y^2
//
// So:
//   log p_y(y) = log_gaussian(x | mu, sigma) - sum_d log(1 - y_d^2)
//
// The epsilon inside log(...) avoids -inf when y is numerically saturated
// at ±1 (which happens often once the policy is confident).
inline torch::Tensor squashed_gaussian_log_prob(const torch::Tensor& raw_action, const torch::Tensor& mean,
                                                const torch::Tensor& log_std) {
    constexpr double kEpsilon = 1e-6;
    const auto raw_log_prob = gaussian_log_prob(raw_action, mean, log_std);
    const auto squashed_action = squash_action(raw_action);
    const auto correction = torch::log(1.0 - torch::pow(squashed_action, 2) + kEpsilon).sum(1, true);
    return raw_log_prob - correction;
}

}  // namespace mujoco_rl_training
