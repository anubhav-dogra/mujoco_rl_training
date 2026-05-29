#pragma once

#include <ATen/core/interned_strings.h>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <envs/EnvTypes.hpp>

namespace mujoco_rl_training {

// =============================================================================
// Action scaling helpers.
//
// Policy networks emit "normalized" actions — typically in [-1, 1] after a
// tanh squash. Environments want actual physical commands (e.g. torques in
// Nm). These helpers do the affine map between the two.
// =============================================================================

// Symmetric scaling: assumes action limits are ±limit per dim. The output
// action_i is simply normalized_i * limit_i. Use this if your action space
// is centered at zero.
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

// General affine map from [-1, 1] to [min_limit, max_limit] per dim:
//
//     action_i = min_i + 0.5 * (normalized_i + 1) * (max_i - min_i)
//
// At normalized = -1, returns min_i; at +1, returns max_i; at 0, the midpoint.
// Use this for asymmetric action ranges (e.g. one-sided actuators).
//
// Note: we deliberately do NOT clip the input. Callers feed `tanh(...)` which
// is already in (-1, 1) by construction, and the env itself clamps to its
// torque limit as a final safety net.
inline std::vector<double> scale_action_from_action_space(const std::vector<double>& normalized_actions,
                                                          const ActionLimitSpace& action_limits) {
    if (normalized_actions.size() != action_limits.size()) {
        throw std::runtime_error("ActioUtils: action size doesn't match with limits for scaling");
    }
    std::vector<double> scaled_action;
    scaled_action.reserve(normalized_actions.size());

    for (std::size_t i = 0; i < normalized_actions.size(); ++i) {
        scaled_action.push_back(action_limits.min_limit[i] + 0.5 * (normalized_actions[i] + 1.0) *
                                                                 (action_limits.max_limit[i] - action_limits.min_limit[i]));
    }
    return scaled_action;
}

}  // namespace mujoco_rl_training
