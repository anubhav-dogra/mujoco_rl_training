#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_core/MujocoSimCore.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace mujoco_rl_training {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

DoublePendulumEnv::DoublePendulumEnv(const DoublePendulumEnvConfig& config)
    : config_(config), sim_core_(nullptr), rng_(config.seed) {
    MujocoSimCore::Config core_config;
    core_config.control_mode = TORQUE;
    core_config.simulation_frequency = config_.simulation_frequency;
    core_config.xml_location = config_.xml_path;
    core_config.visualization_enabled = false;
    if (config_.max_torques.size() != 2) {
        throw std::runtime_error("DoublePendulumEnv: expected exactly 2 max torques.");
    }
    if (config_.joint_names.size() != 2) {
        throw std::runtime_error("DoublePendulumEnv: expected exactly 2 joint names.");
    }
    if (config_.target_angles.size() != config_.joint_names.size()) {
        throw std::runtime_error("DoublePendulumEnv: target_angles size must match joint_names size.");
    }
    if (config_.angle_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("DoublePendulumEnv: angle_cost_weights size must match joint_names size.");
    }
    if (config_.velocity_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("DoublePendulumEnv: velocity_cost_weights size must match joint_names size.");
    }
    if (config_.control_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("DoublePendulumEnv: control_cost_weights size must match joint_names size.");
    }
    sim_core_ = std::make_unique<MujocoSimCore>(core_config);

    pos_indices_.reserve(config_.joint_names.size());
    vel_indices_.reserve(config_.joint_names.size());
    control_indices_.reserve(config_.joint_names.size());

    const auto& sim_joint_state_indices = sim_core_->joint_state_indices_by_name();
    const auto& sim_pos_indices_ = sim_core_->joint_position_indices();
    const auto& sim_vel_indices_ = sim_core_->joint_velocity_indices();
    const auto& sim_control_indices_ = sim_core_->control_indices_by_name();
    for (const auto& joint_name : config_.joint_names) {
        const auto it = sim_joint_state_indices.find(joint_name);
        if (it == sim_joint_state_indices.end()) {
            throw std::runtime_error("Joint name not found: " + joint_name);
        }
        const std::size_t joint_idx = it->second;
        pos_indices_.push_back(sim_pos_indices_[joint_idx]);
        vel_indices_.push_back(sim_vel_indices_[joint_idx]);

        const auto itc = sim_control_indices_.find(joint_name);
        if (itc == sim_control_indices_.end()) {
            throw std::runtime_error("Control joint name not found: " + joint_name);
        }
        control_indices_.push_back(itc->second);
    }
}
DoublePendulumEnv::~DoublePendulumEnv() = default;

std::vector<double> DoublePendulumEnv::reset() {
    std::uniform_real_distribution<double> theta_dist(-config_.reset_angle_range, config_.reset_angle_range);
    std::uniform_real_distribution<double> theta_dot_dist(-config_.reset_velocity_range, config_.reset_velocity_range);

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        sim_core_->reset();
        for (size_t i = 0; i < control_indices_.size(); ++i) {
            sim_core_->set_effort_command(control_indices_[i], 0.0);
        }
        for (size_t i = 0; i < pos_indices_.size(); ++i) {
            sim_core_->data()->qpos[pos_indices_[i]] = theta_dist(rng_);
            sim_core_->data()->qvel[vel_indices_[i]] = theta_dot_dist(rng_);
        }
        mj_forward(sim_core_->model(), sim_core_->data());
    }
    step_count_ = 0;
    return observation();
}

std::vector<double> DoublePendulumEnv::observation() const {
    std::vector<double> theta{}, theta_dot{};
    theta.resize(pos_indices_.size());
    theta_dot.resize(pos_indices_.size());
    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        for (size_t i = 0; i < pos_indices_.size(); ++i) {
            theta[i] = sim_core_->data()->qpos[pos_indices_[i]];
            theta_dot[i] = sim_core_->data()->qvel[vel_indices_[i]];
        }
    }

    return {std::cos(theta[0]), std::sin(theta[0]), theta_dot[0], std::cos(theta[1]), std::sin(theta[1]), theta_dot[1]};
}

MujocoSimCore& DoublePendulumEnv::sim_core() { return *sim_core_; }

const MujocoSimCore& DoublePendulumEnv::sim_core() const { return *sim_core_; }

DoublePendulumStepResult DoublePendulumEnv::step(const std::vector<double>& action) {
    if (action.size() != control_indices_.size()) {
        throw std::runtime_error("DoublePendulumEnv: action size must match joint/control count.");
    }

    std::vector<double> clipped_action{};
    clipped_action.reserve(action.size());
    for (size_t i = 0; i < action.size(); ++i) {
        clipped_action.push_back(std::clamp(action[i], -config_.max_torques[i], config_.max_torques[i]));
    }
    std::vector<double> theta;
    theta.resize(pos_indices_.size());
    std::vector<double> theta_dot;
    theta_dot.resize(vel_indices_.size());

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        for (size_t i = 0; i < control_indices_.size(); ++i) {
            sim_core_->set_effort_command(control_indices_[i], clipped_action[i]);
        }
        for (int i = 0; i < config_.repeat_action; ++i) {
            sim_core_->step();
        }
        for (size_t i = 0; i < pos_indices_.size(); ++i) {
            theta[i] = sim_core_->data()->qpos[pos_indices_[i]];
            theta_dot[i] = sim_core_->data()->qvel[vel_indices_[i]];
        }
    }

    DoublePendulumStepResult result;
    result.reward = compute_reward(theta, theta_dot, clipped_action);
    result.observations = observation();
    ++step_count_;
    result.terminated = false;
    result.truncated = step_count_ >= config_.episode_horizon;
    return result;
}

double DoublePendulumEnv::compute_reward(const std::vector<double>& theta, const std::vector<double>& theta_dot,
                                         const std::vector<double>& action) const {
    if (theta.size() != config_.joint_names.size() || theta_dot.size() != config_.joint_names.size() ||
        action.size() != config_.joint_names.size()) {
        throw std::runtime_error("DoublePendulumEnv: reward input size mismatch.");
    }

    double cost = 0.0;
    for (std::size_t i = 0; i < config_.joint_names.size(); ++i) {
        const double angle_error = normalize_angle(theta[i] - config_.target_angles[i]);
        cost += config_.angle_cost_weights[i] * angle_error * angle_error;
        cost += config_.velocity_cost_weights[i] * theta_dot[i] * theta_dot[i];
        cost += config_.control_cost_weights[i] * action[i] * action[i];
    }

    return -cost;
}

double DoublePendulumEnv::normalize_angle(double angle) const {
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}
}  // namespace mujoco_rl_training
