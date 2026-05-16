#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

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

}  // namespace mujoco_rl_training
