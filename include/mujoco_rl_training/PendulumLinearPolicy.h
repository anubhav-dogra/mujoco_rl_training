#pragma once
#include <array>

namespace mujoco_rl_training {
struct PendulumLinearPolicy {
    double action_from_obs(const std::array<double, 3>& observations) const {
        return (weights[0] * observations[0]) + (weights[1] * observations[1]) + (weights[2] * observations[2]) + bias;
    };

    std::array<double, 3> weights{};
    double bias = 0;
};
}  // namespace mujoco_rl_training
