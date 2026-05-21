#pragma once
#include <cstddef>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <envs/EnvTypes.hpp>

class MujocoSimCore;
namespace mujoco_rl_training {

struct DoublePendulumEnvConfig {
    std::string xml_path{};
    int simulation_frequency = 1000;
    std::vector<double> max_torques{15, 30};  // max torques allowed for each joint
    unsigned int seed = 0;                    // initlaize seed for random generation
    int episode_horizon = 2000;               // how long the episode to run..
    int repeat_action =
        20;  // how many times action to repeat for the same episode so that policy is trained at 1000/20 = 50Hz
    double reset_angle_range = 0.35;    // for both joints
    double reset_velocity_range = 0.5;  // for both joints
    std::vector<std::string> joint_names{};
    std::vector<double> target_angles{};
    std::vector<double> angle_cost_weights{};
    std::vector<double> target_link_angles{};
    std::vector<double> link_angle_cost_weights{};
    std::vector<double> velocity_cost_weights{};
    std::vector<double> control_cost_weights{};
};

class DoublePendulumEnv : public EnvInterface {
   public:
    explicit DoublePendulumEnv(const DoublePendulumEnvConfig& config);
    StepResult step(const std::vector<double>& actions) override;
    std::vector<double> observation() const override;
    std::vector<double> reset() override;
    std::vector<double> reset(const ResetSpace& reset_space) override;
    const EnvSpec& env_spec() const override;
    void set_training_weights(const std::vector<double>& angle_cost_weights,
                              const std::vector<double>& velocity_cost_weights,
                              const std::vector<double>& control_cost_weights, const std::vector<double>& max_torques,
                              const std::vector<double>& link_angle_cost_weights = {});
    const DoublePendulumEnvConfig& config() const;
    MujocoSimCore& sim_core();
    const MujocoSimCore& sim_core() const;

    ~DoublePendulumEnv();

   private:
    double compute_reward(const std::vector<double>& theta, const std::vector<double>& theta_dot,
                          const std::vector<double>& action) const;
    double normalize_angle(double angle) const;
    DoublePendulumEnvConfig config_;
    EnvSpec env_spec_;
    std::unique_ptr<MujocoSimCore> sim_core_;
    std::vector<std::size_t> pos_indices_;
    std::vector<std::size_t> vel_indices_;
    std::vector<std::size_t> control_indices_;
    std::mt19937 rng_;
    int step_count_ = 0;
};

}  // namespace mujoco_rl_training
