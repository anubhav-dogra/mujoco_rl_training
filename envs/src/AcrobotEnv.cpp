#include <envs/AcrobotEnv.hpp>
#include <mujoco_core/MujocoSimCore.h>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <iostream>
namespace mujoco_rl_training {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

AcrobotEnv::AcrobotEnv(const AcrobotEnvConfig& config) : config_(config), sim_core_(nullptr), rng_(config.seed) {
    MujocoSimCore::Config core_config;
    core_config.control_mode = TORQUE;
    core_config.simulation_frequency = config_.simulation_frequency;
    core_config.xml_location = config_.xml_path;
    core_config.visualization_enabled = false;
    core_config.tracked_frame_names = config_.tracked_frame_names;
    if (config_.max_torques.size() != 1) {
        throw std::runtime_error("AcrobotEnv: expected exactly 1 max torque.");
    }
    if (config_.joint_names.size() != 2) {
        throw std::runtime_error("AcrobotEnv: expected exactly 2 joint names.");
    }
    if (config_.target_angles.size() != config_.joint_names.size()) {
        throw std::runtime_error("AcrobotEnv: target_angles size must match joint_names size.");
    }
    if (config_.angle_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("AcrobotEnv: angle_cost_weights size must match joint_names size.");
    }
    if (!config_.target_link_angles.empty() && config_.target_link_angles.size() != config_.joint_names.size()) {
        throw std::runtime_error("AcrobotEnv: target_link_angles size must match joint_names size.");
    }
    if (!config_.link_angle_cost_weights.empty() &&
        config_.link_angle_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("AcrobotEnv: link_angle_cost_weights size must match joint_names size.");
    }
    if (config_.velocity_cost_weights.size() != config_.joint_names.size()) {
        throw std::runtime_error("AcrobotEnv: velocity_cost_weights size must match joint_names size.");
    }
    if (config_.control_cost_weights.size() != 1) {
        throw std::runtime_error("AcrobotEnv: control_cost_weights size must be 1.");
    }
    if (config_.tracked_frame_names.size() != 1) {
        throw std::runtime_error("Missing tracked_frame_names");
    }
    sim_core_ = std::make_unique<MujocoSimCore>(core_config);

    pos_indices_.reserve(config_.joint_names.size());
    vel_indices_.reserve(config_.joint_names.size());
    env_spec_.action_dim = 1;
    env_spec_.observation_dim = 6;
    env_spec_.action_limit_space.max_limit = {config_.max_torques[0]};
    env_spec_.action_limit_space.min_limit = {-config_.max_torques[0]};
    control_indices_.reserve(env_spec_.action_dim);

    const auto& sim_joint_state_indices = sim_core_->joint_state_indices_by_name();
    const auto& sim_pos_indices_ = sim_core_->joint_position_indices();
    const auto& sim_vel_indices_ = sim_core_->joint_velocity_indices();
    const auto& sim_control_indices = sim_core_->control_indices_by_name();
    for (const auto& joint_name : config_.joint_names) {
        const auto it = sim_joint_state_indices.find(joint_name);
        if (it == sim_joint_state_indices.end()) {
            throw std::runtime_error("Joint name not found: " + joint_name);
        }
        const std::size_t joint_idx = it->second;
        pos_indices_.push_back(sim_pos_indices_[joint_idx]);
        vel_indices_.push_back(sim_vel_indices_[joint_idx]);
    }

    const auto control_it = sim_control_indices.find(config_.control_name);
    if (control_it == sim_control_indices.end()) {
        throw std::runtime_error("Control name not found: " + config_.control_name);
    }
    control_indices_.push_back(control_it->second);

    // setting the target energy for reward:
    // Target of acrobot standing upwards:
    // q1 = kPi, q2 = 0;
    sim_core_->data()->qpos[pos_indices_[0]] = kPi;
    sim_core_->data()->qpos[pos_indices_[1]] = 0.0;

    sim_core_->data()->qvel[vel_indices_[0]] = 0.0;
    sim_core_->data()->qvel[vel_indices_[1]] = 0.0;
    mj_forward(sim_core_->model(), sim_core_->data());  // recomputes positions/velocities in world frame
    mj_energyPos(sim_core_->model(), sim_core_->data());
    mj_energyVel(sim_core_->model(), sim_core_->data());
    target_energy = sim_core_->data()->energy[1] + sim_core().data()->energy[0];
    std::cout << "target_energy: = " << target_energy << std::endl;
}
AcrobotEnv::~AcrobotEnv() = default;

const EnvSpec& AcrobotEnv::env_spec() const { return env_spec_; }

std::vector<double> AcrobotEnv::reset() {
    ResetSpace reset_space;
    reset_space.qpos_centers = std::vector<double>(config_.joint_names.size(), 0.0);
    reset_space.qpos_ranges = std::vector<double>(config_.joint_names.size(), config_.reset_angle_range);
    reset_space.qvel_centers = std::vector<double>(config_.joint_names.size(), 0.0);
    reset_space.qvel_ranges = std::vector<double>(config_.joint_names.size(), config_.reset_velocity_range);
    return reset(reset_space);
}

std::vector<double> AcrobotEnv::reset(const ResetSpace& reset_space) {
    if (reset_space.qpos_ranges.size() != pos_indices_.size() ||
        reset_space.qvel_ranges.size() != vel_indices_.size() ||
        reset_space.qvel_centers.size() != vel_indices_.size() ||
        reset_space.qpos_centers.size() != pos_indices_.size()) {
        throw std::runtime_error("Size mismatch in reset call");
    }

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        sim_core_->reset();
        for (size_t i = 0; i < control_indices_.size(); ++i) {
            sim_core_->set_effort_command(control_indices_[i], 0.0);
        }
        for (size_t i = 0; i < pos_indices_.size(); ++i) {
            std::uniform_real_distribution<double> qpos_dist(reset_space.qpos_centers[i] - reset_space.qpos_ranges[i],
                                                             reset_space.qpos_centers[i] + reset_space.qpos_ranges[i]);
            std::uniform_real_distribution<double> qvel_dist(reset_space.qvel_centers[i] - reset_space.qvel_ranges[i],
                                                             reset_space.qvel_centers[i] + reset_space.qvel_ranges[i]);
            sim_core_->data()->qpos[pos_indices_[i]] = qpos_dist(rng_);
            sim_core_->data()->qvel[vel_indices_[i]] = qvel_dist(rng_);
        }
        mj_forward(sim_core_->model(), sim_core_->data());
    }
    step_count_ = 0;
    return observation();
}

std::vector<double> AcrobotEnv::observation() const {
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

    const double link_2_angle = theta[0] + theta[1];
    return {std::cos(theta[0]),     std::sin(theta[0]), std::cos(link_2_angle),
            std::sin(link_2_angle), theta_dot[0],       theta_dot[1]};
}

MujocoSimCore& AcrobotEnv::sim_core() { return *sim_core_; }

const MujocoSimCore& AcrobotEnv::sim_core() const { return *sim_core_; }

const AcrobotEnvConfig& AcrobotEnv::config() const { return config_; }

void AcrobotEnv::set_training_weights(const std::vector<double>& angle_cost_weights,
                                      const std::vector<double>& velocity_cost_weights,
                                      const std::vector<double>& control_cost_weights,
                                      const std::vector<double>& max_torques,
                                      const std::vector<double>& link_angle_cost_weights) {
    const auto expected_size = config_.joint_names.size();
    if (angle_cost_weights.size() != expected_size || velocity_cost_weights.size() != expected_size) {
        throw std::runtime_error("AcrobotEnv: angle and velocity training weight vectors must match joint count.");
    }
    if (control_cost_weights.size() != control_indices_.size() || max_torques.size() != control_indices_.size()) {
        throw std::runtime_error("AcrobotEnv: control weight and max torque vectors must match control count.");
    }
    if (!link_angle_cost_weights.empty() && link_angle_cost_weights.size() != expected_size) {
        throw std::runtime_error("AcrobotEnv: link angle cost weights must match joint count.");
    }

    config_.angle_cost_weights = angle_cost_weights;
    config_.velocity_cost_weights = velocity_cost_weights;
    config_.control_cost_weights = control_cost_weights;
    config_.max_torques = max_torques;
    if (!link_angle_cost_weights.empty()) {
        config_.link_angle_cost_weights = link_angle_cost_weights;
    }
    env_spec_.action_limit_space.max_limit = {config_.max_torques[0]};
    env_spec_.action_limit_space.min_limit = {-config_.max_torques[0]};
}

StepResult AcrobotEnv::step(const std::vector<double>& action) {
    if (action.size() != control_indices_.size()) {
        throw std::runtime_error("AcrobotEnv: action size must match joint/control count.");
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
        mj_energyPos(sim_core_->model(), sim_core_->data());
        mj_energyVel(sim_core_->model(), sim_core_->data());

        for (size_t i = 0; i < pos_indices_.size(); ++i) {
            theta[i] = sim_core_->data()->qpos[pos_indices_[i]];
            theta_dot[i] = sim_core_->data()->qvel[vel_indices_[i]];
        }
    }

    StepResult result;
    result.reward = compute_reward(theta, theta_dot, clipped_action);
    result.observation = observation();
    ++step_count_;
    result.terminated = false;
    result.truncated = step_count_ >= config_.episode_horizon;
    return result;
}

double AcrobotEnv::compute_reward(const std::vector<double>& theta, const std::vector<double>& theta_dot,
                                  const std::vector<double>& action) const {
    if (theta.size() != config_.joint_names.size() || theta_dot.size() != config_.joint_names.size() ||
        action.size() != config_.max_torques.size()) {
        throw std::runtime_error("AcrobotEnv: reward input size mismatch.");
    }

    // ---------------------------------------------------------------------------
    // Phase-blended reward:
    //   Far from upright (link-angle errors big)  -> "swing-up" phase: only energy mismatch matters.
    //   Near upright (errors small)               -> "balance" phase: state-quadratic (LQR-style).
    // A smooth Gaussian "phase" factor blends the two. Plus a sparse bonus when
    // truly balanced (small angle errors AND small velocities) gives the policy
    // a clear cliff in the value landscape to climb.
    // ---------------------------------------------------------------------------

    // Step 1: compute per-link angle errors and their cumulative norm.
    double link_angle = 0.0;
    double total_link_err_sq = 0.0;
    std::vector<double> link_errs(config_.joint_names.size(), 0.0);
    for (std::size_t i = 0; i < config_.joint_names.size(); ++i) {
        link_angle += theta[i];
        link_errs[i] = normalize_angle(link_angle - config_.target_link_angles[i]);
        total_link_err_sq += link_errs[i] * link_errs[i];
    }
    const double link_err_norm = std::sqrt(total_link_err_sq);

    // Step 2: phase blend. balance_weight = 1 at upright, ~0 far away.
    const double sigma = config_.phase_blend_sigma;
    const double balance_weight = std::exp(-link_err_norm * link_err_norm / (sigma * sigma));
    const double swingup_weight = 1.0 - balance_weight;

    double cost = 0.0;

    // Step 3: swing-up cost (energy mismatch). Fades out near upright.
    const double current_energy = sim_core_->data()->energy[0] + sim_core_->data()->energy[1];
    const double energy_err = current_energy - target_energy;
    cost += swingup_weight * config_.energy_weight * energy_err * energy_err;

    // Step 4: balance cost (state-quadratic). Fades in near upright.
    for (std::size_t i = 0; i < config_.joint_names.size(); ++i) {
        cost += balance_weight * config_.link_angle_cost_weights[i] * link_errs[i] * link_errs[i];
        cost += balance_weight * config_.velocity_cost_weights[i] * theta_dot[i] * theta_dot[i];
    }

    // Step 5: sparse bonus for being "actually balanced" — a positive reward (subtracted from cost).
    // Creates a discrete cliff that the policy can feel even when quadratic gradients are near zero.
    const bool angles_in_sweet_spot = link_err_norm < config_.balance_radius;
    bool velocities_in_sweet_spot = true;
    for (std::size_t i = 0; i < config_.joint_names.size(); ++i) {
        if (std::abs(theta_dot[i]) > config_.balance_velocity_max) {
            velocities_in_sweet_spot = false;
            break;
        }
    }
    if (angles_in_sweet_spot && velocities_in_sweet_spot) {
        cost -= config_.balance_bonus;
    }

    // Step 6: control cost (always — tiny disincentive against wild torques).
    for (std::size_t i = 0; i < action.size(); ++i) {
        cost += config_.control_cost_weights[i] * action[i] * action[i];
    }

    return -cost;
}

double AcrobotEnv::normalize_angle(double angle) const {
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}
}  // namespace mujoco_rl_training
