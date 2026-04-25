#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_rl_training/DoublePendulumLinearPolicy.h>
#include <mujoco_rl_training/DoublePendulumPolicyMetadata.h>
#include <mujoco_rl_training/VisualDemoUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kDefaultPolicyArtifactPath = "artifacts/double_pendulum_best_policy.txt";
const char* kLegacyPolicyArtifactPath = "artifacts/double_pendulum_random_search_policy.txt";

struct LoadedPolicy {
    mujoco_rl_training::DoublePendulumLinearPolicy policy;
    bool has_sigma = false;
    std::vector<double> sigma{};
};

LoadedPolicy load_policy(const std::string& policy_path) {
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

    if (values.size() != 14 && values.size() != 16) {
        throw std::runtime_error("Expected 14-value linear or 16-value Gaussian double pendulum policy artifact: " +
                                 policy_path);
    }

    LoadedPolicy loaded_policy;
    std::size_t value_index = 0;
    for (auto& row : loaded_policy.policy.weights) {
        for (double& weight : row) {
            weight = values[value_index++];
        }
    }
    loaded_policy.policy.bias[0] = values[value_index++];
    loaded_policy.policy.bias[1] = values[value_index++];

    if (values.size() == 16) {
        loaded_policy.has_sigma = true;
        loaded_policy.sigma = {values[value_index++], values[value_index++]};
    }

    return loaded_policy;
}

}  // namespace

int main(int argc, char* argv[]) {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.max_torques = {40.0, 30.0};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {3.0, 1.5};
    config.velocity_cost_weights = {0.15, 0.08};
    config.control_cost_weights = {0.0001, 0.0001};
    config.episode_horizon = 500;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;
    config.reset_angle_range = 0.35;
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

    const std::string metadata_path =
        (argc > 2) ? argv[2] : mujoco_rl_training::double_pendulum_metadata_path_for_policy(policy_path);
    const auto metadata = mujoco_rl_training::load_double_pendulum_policy_metadata(metadata_path);
    if (metadata.has_value()) {
        config = metadata->config;
        std::cout << "Loaded policy metadata from: " << metadata_path << '\n'
                  << "Saved policy best_return: " << metadata->best_return << '\n'
                  << "Training iterations: " << metadata->num_iterations << '\n'
                  << "Episodes per evaluation: " << metadata->episodes_per_evaluation << '\n'
                  << "Noise std dev: " << metadata->noise_std_dev << std::endl;
    } else {
        std::cout << "Policy metadata not found, using rollout defaults: " << metadata_path << std::endl;
    }

    const auto loaded_policy = load_policy(policy_path);
    const auto& policy = loaded_policy.policy;

    mujoco_rl_training::DoublePendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation:";
    for (double value_obs : observation) {
        std::cout << ' ' << value_obs;
    }
    std::cout << "\nLoaded policy from: " << policy_path << std::endl;
    if (loaded_policy.has_sigma) {
        std::cout << "Policy artifact type: Gaussian mean-policy replay" << " sigma=[" << loaded_policy.sigma[0] << ", "
                  << loaded_policy.sigma[1] << "]" << std::endl;
    } else {
        std::cout << "Policy artifact type: deterministic linear policy" << std::endl;
    }

    return mujoco_rl_training::run_visual_demo(env, observation, [&]() { return policy.action_from_obs(observation); },
                                               {{0.0, 0.0, 0.5}, 3.5, 90.0, 0.0});
}
