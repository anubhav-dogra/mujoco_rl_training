#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

inline torch::Tensor gaussian_log_prob(const torch::Tensor& action, const torch::Tensor& mean,
                                       const torch::Tensor& log_std) {
    constexpr double kLogTwoPi = 1.8378770664093453;
    const auto std = torch::exp(log_std);
    const auto log_prob_per_dim =
        -0.5 * (torch::pow((action - mean) / std, 2) + 2.0 * log_std + kLogTwoPi);
    return log_prob_per_dim.sum(1, true);
}

inline torch::Tensor sample_gaussian_action(const torch::Tensor& mean, const torch::Tensor& log_std) {
    return mean + torch::exp(log_std) * torch::randn_like(mean);
}

inline torch::Tensor squash_action(const torch::Tensor& raw_action) {
    return torch::tanh(raw_action);
}

inline torch::Tensor squashed_gaussian_log_prob(const torch::Tensor& raw_action, const torch::Tensor& mean,
                                                const torch::Tensor& log_std) {
    constexpr double kEpsilon = 1e-6;
    const auto raw_log_prob = gaussian_log_prob(raw_action, mean, log_std);
    const auto squashed_action = squash_action(raw_action);
    const auto correction = torch::log(1.0 - torch::pow(squashed_action, 2) + kEpsilon).sum(1, true);
    return raw_log_prob - correction;
}

}  // namespace mujoco_rl_training
