#pragma once

#include <mujoco_rl_training/PolicyIO.h>
#include <mujoco_rl_training/policies/DoublePendulumLinearPolicy.h>
#include <mujoco_rl_training/policies/PendulumGaussianPolicy.h>
#include <mujoco_rl_training/policies/PendulumLinearPolicy.h>

#include <iomanip>
#include <string>

namespace mujoco_rl_training {

inline void save_pendulum_linear_policy(const std::string& path, const PendulumLinearPolicy& policy) {
    auto output = open_artifact_output(path);
    output << policy.weights[0] << ' ' << policy.weights[1] << ' ' << policy.weights[2] << ' ' << policy.bias << '\n';
}

inline void save_pendulum_gaussian_policy(const std::string& path, const PendulumGaussianPolicy& policy) {
    auto output = open_artifact_output(path);
    output << policy.weights[0] << ' ' << policy.weights[1] << ' ' << policy.weights[2] << ' ' << policy.bias << ' '
           << policy.sigma << '\n';
}

inline void save_double_pendulum_linear_policy(const std::string& path,
                                               const DoublePendulumLinearPolicy& policy) {
    auto output = open_artifact_output(path);
    output << std::setprecision(17);
    for (const auto& row : policy.weights) {
        for (double weight : row) {
            output << weight << ' ';
        }
    }
    output << policy.bias[0] << ' ' << policy.bias[1] << '\n';
}

}  // namespace mujoco_rl_training
