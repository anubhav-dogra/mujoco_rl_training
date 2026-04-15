#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <random>
#include <string>

class MujocoSimCore;

namespace mujoco_rl_training {

struct PendulumEnvConfig {
    std::string xml_path;
    int simulation_frequency = 1000;
    double max_torque = 20.0;
    int episode_horizon = 1000;
    unsigned int seed = 0;
    int repeat_action = 20;
    double reset_angle_range = 0.35;
    double reset_velocity_range = 0.5;
};
struct PendulumStepResult {
    std::array<double, 3> observation{};
    double reward = 0.0;
    bool terminated = false;
    bool truncated = false;
};

class PendulumEnv {
   public:
    explicit PendulumEnv(const PendulumEnvConfig& config);
    std::array<double, 3> reset();
    PendulumStepResult step(double action);
    std::array<double, 3> observation() const;
    MujocoSimCore& sim_core();
    const MujocoSimCore& sim_core() const;

    ~PendulumEnv();

   private:
    PendulumEnvConfig config_;
    std::unique_ptr<MujocoSimCore> sim_core_;
    std::size_t qpos_index_ = 0;
    std::size_t qvel_index_ = 0;
    std::size_t control_index_ = 0;
    int step_count_ = 0;
    std::mt19937 rng_;
    double normalize_angle(double angle) const;
    double current_theta() const;
    double current_theta_dot() const;
};

}  // namespace mujoco_rl_training
