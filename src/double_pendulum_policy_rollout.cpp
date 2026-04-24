#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_rl_training/DoublePendulumLinearPolicy.h>
#include <mujoco_rl_training/VisualDemoUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kDefaultPolicyArtifactPath = "artifacts/double_pendulum_best_policy.txt";
const char* kLegacyPolicyArtifactPath = "artifacts/double_pendulum_random_search_policy.txt";

mujoco_rl_training::DoublePendulumLinearPolicy load_policy(const std::string& policy_path) {
    std::ifstream input(policy_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open saved policy artifact: " + policy_path);
    }

    std::vector<double> values;
    double value = 0.0;
    while (input >> value) {
        values.push_back(value);
    }

    if (!input.eof()) {
        throw std::runtime_error("Failed while parsing saved policy artifact: " + policy_path);
    }

    if (values.size() != 14) {
        throw std::runtime_error("Expected 14 values in double pendulum policy artifact: " + policy_path);
    }

    mujoco_rl_training::DoublePendulumLinearPolicy policy;
    std::size_t value_index = 0;
    for (auto& row : policy.weights) {
        for (double& weight : row) {
            weight = values[value_index++];
        }
    }
    policy.bias[0] = values[value_index++];
    policy.bias[1] = values[value_index++];

    return policy;
}

}  // namespace

int main(int argc, char* argv[]) {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.max_torques = {40.0, 30.0};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {1.0, 1.0};
    config.velocity_cost_weights = {0.05, 0.02};
    config.control_cost_weights = {0.001, 0.001};
    config.episode_horizon = 400;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;
    config.reset_angle_range = 2.0;
    config.reset_velocity_range = 0.5;

    std::string policy_path = (argc > 1) ? argv[1] : kDefaultPolicyArtifactPath;
    if (argc <= 1) {
        std::ifstream primary_check(policy_path);
        if (!primary_check.is_open()) {
            std::ifstream legacy_check(kLegacyPolicyArtifactPath);
            if (legacy_check.is_open()) {
                policy_path = kLegacyPolicyArtifactPath;
            }
        }
    }
    const auto policy = load_policy(policy_path);

    mujoco_rl_training::DoublePendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation:";
    for (double value_obs : observation) {
        std::cout << ' ' << value_obs;
    }
    std::cout << "\nLoaded policy from: " << policy_path << std::endl;

    return mujoco_rl_training::run_visual_demo(
        env, observation, [&]() { return policy.action_from_obs(observation); }, {{0.0, 0.0, -0.6}, 3.0, 135.0, -20.0});
}
