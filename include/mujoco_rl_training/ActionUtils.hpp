#pragma once

#include <ATen/core/interned_strings.h>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <envs/EnvTypes.hpp>

namespace mujoco_rl_training {

/* Scale Action from Normalized to Actual Action
 * Example: normalized torque to the Actual Torque for Env Physics
 */

inline std::vector<double> scale_action(const std::vector<double>& normalized_actions,
                                        const std::vector<double>& action_limits) {
    if (normalized_actions.size() != action_limits.size()) {
        throw std::runtime_error("Scale_action: Normalized Action Size is not equal to Action Limits");
    }
    std::vector<double> actions;
    actions.reserve(normalized_actions.size());

    for (std::size_t i = 0; i < normalized_actions.size(); ++i) {
        actions.push_back(normalized_actions[i] * action_limits[i]);
    }

    return actions;
}

// More generic
inline std::vector<double> scale_action_from_action_space(const std::vector<double>& normalized_actions,
                                                          const ActionLimitSpace& action_limits) {
    if (normalized_actions.size() != action_limits.size()) {
        throw std::runtime_error("ActioUtils: action size doesn't match with limits for scaling");
    }
    std::vector<double> scaled_action;
    scaled_action.reserve(normalized_actions.size());

    for (std::size_t i = 0; i < normalized_actions.size(); ++i) {
        const double clipped = std::clamp(normalized_actions[i], -1.0, 1.0);
        scaled_action.push_back(action_limits.min_limit[i] +
                                0.5 * (clipped + 1.0) * (action_limits.max_limit[i] - action_limits.min_limit[i]));
    }
    return scaled_action;
}

}  // namespace mujoco_rl_training
