#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mujoco_rl_training {

template <typename Trajectory>
inline void compute_gae(Trajectory& trajectory, double gamma, double lambda) {
    trajectory.advantages.assign(trajectory.size(), 0.0);
    trajectory.returns.assign(trajectory.size(), 0.0);

    double gae = 0.0;
    double next_value = trajectory.bootstrap_value;
    for (int t = static_cast<int>(trajectory.size()) - 1; t >= 0; --t) {
        const std::size_t index = static_cast<std::size_t>(t);
        const double mask = trajectory.dones[index] ? 0.0 : 1.0;
        const double delta = trajectory.rewards[index] + (gamma * next_value * mask) - trajectory.values[index];
        gae = delta + gamma * lambda * mask * gae;

        trajectory.advantages[index] = gae;
        trajectory.returns[index] = gae + trajectory.values[index];
        next_value = trajectory.values[index];
    }
}

template <typename Trajectory>
inline void normalize_advantages(Trajectory& trajectory) {
    if (trajectory.advantages.empty()) {
        return;
    }

    double sum = 0.0;
    double sq_sum = 0.0;
    for (double advantage : trajectory.advantages) {
        sum += advantage;
        sq_sum += advantage * advantage;
    }

    const double count = static_cast<double>(trajectory.advantages.size());
    const double mean = sum / count;
    const double variance = std::max((sq_sum / count) - (mean * mean), 1e-8);
    const double stddev = std::sqrt(variance);

    for (double& advantage : trajectory.advantages) {
        advantage = (advantage - mean) / stddev;
    }
}

}  // namespace mujoco_rl_training
