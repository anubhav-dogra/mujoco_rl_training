#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace mujoco_rl_training {
struct DoublePendulumLinearPolicy {
    std::vector<double> action_from_obs(const std::vector<double> &observations) const {
        if (observations.size() != 6) {
            throw std::runtime_error("observations size is not as double Pendulum");
        }
        std::vector<double> actions(2, 0.0);
        actions[0] = bias[0];
        actions[1] = bias[1];
        for (std::size_t i = 0; i < observations.size(); ++i) {
            actions[0] += weights[0][i] * observations[i];
            actions[1] += weights[1][i] * observations[i];
        }
        return actions;
    };

    std::array<std::array<double, 6>, 2> weights{};
    std::array<double, 2> bias{};
};
}  // namespace mujoco_rl_training
