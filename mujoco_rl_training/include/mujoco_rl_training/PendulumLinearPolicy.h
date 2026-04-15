#pragma once
#include <array>

namespace mujoco_rl_training {
struct PendulumLinearPolicy {
    double action_from_obs(const std::array<double, 3>& observations) const;

    std::array<double, 3> weights{};
    double bias = 0;
};
}  // namespace mujoco_rl_training
