#include <mujoco_rl_training/PendulumEnv.h>
#include <mujoco_core/MujocoSimCore.h>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <random>
#include <stdexcept>

namespace mujoco_rl_training {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

PendulumEnv::PendulumEnv(const PendulumEnvConfig& config) : config_(config), sim_core_(nullptr), rng_(config.seed) {
    MujocoSimCore::Config core_config;
    core_config.xml_location = config_.xml_path;
    core_config.control_mode = TORQUE;
    core_config.simulation_frequency = config_.simulation_frequency;
    core_config.visualization_enabled = false;
    sim_core_ = std::make_unique<MujocoSimCore>(core_config);

    const auto& joint_state_indices = sim_core_->joint_state_indices_by_name();
    const auto& qpos_indices = sim_core_->joint_position_indices();
    const auto& qvel_indices = sim_core_->joint_velocity_indices();
    const auto& control_indices = sim_core_->control_indices_by_name();

    const auto joint_it = joint_state_indices.find("pendulum_joint");
    if (joint_it == joint_state_indices.end()) {
        throw std::runtime_error("PendulumEnv: 'pendulum_joint' not found in MuJoCo joint state map.");
    }
    const std::size_t joint_index = joint_it->second;
    qpos_index_ = qpos_indices[joint_index];
    qvel_index_ = qvel_indices[joint_index];

    const auto control_it = control_indices.find("pendulum_joint");
    if (control_it == control_indices.end()) {
        throw std::runtime_error("PendulumEnv: 'pendulum_joint' not found in MuJoCo control map.");
    }

    control_index_ = control_it->second;
}

PendulumEnv::~PendulumEnv() = default;

std::array<double, 3> PendulumEnv::reset() {
    std::uniform_real_distribution<double> theta_dist(-1, 1);
    std::uniform_real_distribution<double> theta_dot_dist(-1, 1);

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        sim_core_->reset();

        sim_core_->set_effort_command(control_index_, 0.0);

        sim_core_->data()->qpos[qpos_index_] = kPi + theta_dist(rng_);
        sim_core_->data()->qvel[qvel_index_] = theta_dot_dist(rng_);

        mj_forward(sim_core_->model(), sim_core_->data());
    }
    step_count_ = 0;
    return observation();
}

PendulumStepResult PendulumEnv::step(double action) {
    const double clipped_action = std::clamp(action, -config_.max_torque, config_.max_torque);
    double theta = 0.0;
    double theta_dot = 0.0;

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        sim_core_->set_effort_command(control_index_, clipped_action);
        for (int i = 0; i < config_.repeat_action; ++i) {
            sim_core_->step();
        }
        theta = sim_core_->data()->qpos[qpos_index_];
        theta_dot = sim_core_->data()->qvel[qvel_index_];
    }

    const double theta_normalized = normalize_angle(theta);

    PendulumStepResult result;
    result.observation = {std::cos(theta), std::sin(theta), theta_dot};
    result.reward = -((theta_normalized * theta_normalized) + (0.1 * theta_dot * theta_dot) +
                      (0.001 * clipped_action * clipped_action));
    ++step_count_;
    result.terminated = false;
    result.truncated = step_count_ >= config_.episode_horizon;
    return result;
}

std::array<double, 3> PendulumEnv::observation() const {
    double theta = 0.0;
    double theta_dot = 0.0;

    {
        std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
        theta = sim_core_->data()->qpos[qpos_index_];
        theta_dot = sim_core_->data()->qvel[qvel_index_];
    }

    return {std::cos(theta), std::sin(theta), theta_dot};
}

MujocoSimCore& PendulumEnv::sim_core() { return *sim_core_; }

const MujocoSimCore& PendulumEnv::sim_core() const { return *sim_core_; }

double PendulumEnv::normalize_angle(double angle) const {
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}

double PendulumEnv::current_theta() const {
    std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
    return sim_core_->data()->qpos[qpos_index_];
}

double PendulumEnv::current_theta_dot() const {
    std::lock_guard<std::recursive_mutex> lock(sim_core_->state_mutex());
    return sim_core_->data()->qvel[qvel_index_];
}

}  // namespace mujoco_rl_training
