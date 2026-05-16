#pragma once

#include <mujoco_rl_training/policies/DoublePendulumLinearPolicy.h>
#include <mujoco_rl_training/policies/PendulumLinearPolicy.h>

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mujoco_rl_training {

struct LoadedPendulumPolicy {
    PendulumLinearPolicy policy;
    bool has_sigma = false;
    double sigma = 0.0;
};

struct LoadedDoublePendulumPolicy {
    DoublePendulumLinearPolicy policy;
    bool has_sigma = false;
    std::vector<double> sigma{};
};

inline std::vector<double> load_policy_values(const std::string& policy_path) {
    std::ifstream input(policy_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open saved policy artifact: " + policy_path);
    }

    std::vector<double> values;
    double value = 0.0;
    while (input >> value) {
        values.push_back(value);
    }

    if (!input.eof()) {
        throw std::runtime_error("Failed while parsing saved policy artifact: " + policy_path);
    }

    return values;
}

inline LoadedPendulumPolicy load_pendulum_policy(const std::string& policy_path) {
    const auto values = load_policy_values(policy_path);
    if (values.size() != 4 && values.size() != 5) {
        throw std::runtime_error("Expected 4-value linear policy or 5-value Gaussian policy artifact: " + policy_path);
    }

    LoadedPendulumPolicy loaded_policy;
    loaded_policy.policy.weights[0] = values[0];
    loaded_policy.policy.weights[1] = values[1];
    loaded_policy.policy.weights[2] = values[2];
    loaded_policy.policy.bias = values[3];
    if (values.size() == 5) {
        loaded_policy.has_sigma = true;
        loaded_policy.sigma = values[4];
    }

    return loaded_policy;
}

inline LoadedDoublePendulumPolicy load_double_pendulum_policy(const std::string& policy_path) {
    const auto values = load_policy_values(policy_path);
    if (values.size() != 14 && values.size() != 16) {
        throw std::runtime_error("Expected 14-value linear or 16-value Gaussian double pendulum policy artifact: " +
                                 policy_path);
    }

    LoadedDoublePendulumPolicy loaded_policy;
    std::size_t value_index = 0;
    for (auto& row : loaded_policy.policy.weights) {
        for (double& weight : row) {
            weight = values[value_index++];
        }
    }
    loaded_policy.policy.bias[0] = values[value_index++];
    loaded_policy.policy.bias[1] = values[value_index++];

    if (values.size() == 16) {
        loaded_policy.has_sigma = true;
        loaded_policy.sigma = {values[value_index++], values[value_index++]};
    }

    return loaded_policy;
}

}  // namespace mujoco_rl_training
