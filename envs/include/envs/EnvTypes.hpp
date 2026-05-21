#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <random>
namespace mujoco_rl_training {

struct ActionLimitSpace {
    std::vector<double> max_limit{};
    std::vector<double> min_limit{};

    std::size_t size() const {
        if (min_limit.size() != max_limit.size()) {
            throw std::runtime_error("EnvType: ActionLimitSpace size not matching with min and max");
        }
        return min_limit.size();
    }
};

struct EnvSpec {
    std::size_t observation_dim = 0;
    std::size_t action_dim = 0;
    ActionLimitSpace action_limit_space;
};

// structure of what step function should return...
struct StepResult {
    std::vector<double> observation{};
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
};

struct ResetSpace {
    std::vector<double> qpos_centers{};
    std::vector<double> qpos_ranges{};
    std::vector<double> qvel_centers{};
    std::vector<double> qvel_ranges{};
};

class EnvInterface {
   public:
    virtual ~EnvInterface() = default;
    virtual const EnvSpec& env_spec() const = 0;
    virtual std::vector<double> reset() = 0;
    virtual std::vector<double> reset(const ResetSpace& reset_space) = 0;
    virtual StepResult step(const std::vector<double>& actions) = 0;
    virtual std::vector<double> observation() const = 0;
};

template <typename Env>
std::vector<double> reset_from_scenario(Env& env, const ResetSpace& reset_space) {
    return env.reset(reset_space);
}

template <typename Env>
std::vector<double> reset_from_random_scenario(Env& env, const std::vector<ResetSpace>& reset_spaces,
                                               std::mt19937& rng) {
    if (reset_spaces.empty()) {
        return env.reset();
    }

    std::uniform_int_distribution<std::size_t> scenario_dist(0, reset_spaces.size() - 1);
    return reset_from_scenario(env, reset_spaces[scenario_dist(rng)]);
}
}  // namespace mujoco_rl_training
