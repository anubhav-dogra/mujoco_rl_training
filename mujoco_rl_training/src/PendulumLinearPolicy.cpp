#include <mujoco_rl_training/PendulumLinearPolicy.h>
double mujoco_rl_training::PendulumLinearPolicy::action_from_obs(const std::array<double, 3>& obs) const {
    return (weights[0] * obs[0]) + (weights[1] * obs[1]) + (weights[2] * obs[2]) + bias;
}
