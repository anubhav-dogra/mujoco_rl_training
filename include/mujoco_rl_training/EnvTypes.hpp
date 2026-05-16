#pragma once

#include <vector>

namespace mujoco_rl_training {

struct StepResult {
    std::vector<double> observation{};
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
};

}  // namespace mujoco_rl_training
