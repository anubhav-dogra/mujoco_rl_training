#include <envs/DoublePendulumEnv.h>
#include <mujoco_rl_training/DoublePendulumPolicyMetadata.h>
#include <mujoco_rl_training/RolloutUtils.h>
#include <mujoco_rl_training/artifacts/PolicyArtifacts.h>
#include <mujoco_rl_training/policies/DoublePendulumLinearPolicy.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cmath>
#include <iostream>
#include <random>

namespace {
const char* kPolicyArtifactPath = "artifacts/double_pendulum_best_policy.txt";

const auto action_adapter = [](const auto& policy, const auto& observation, auto&) {
    return policy.action_from_obs(observation);
};
}  // namespace

int main() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.target_angles = {M_PI, 0.0};
    config.angle_cost_weights = {3.0, 1.5};
    config.velocity_cost_weights = {0.25, 0.12};
    config.control_cost_weights = {0.0001, 0.0001};
    config.max_torques = {40, 30};
    config.episode_horizon = 400;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;

    mujoco_rl_training::DoublePendulumEnv env(config);
    mujoco_rl_training::DoublePendulumLinearPolicy best_policy{};

    constexpr int kNumIterations = 1000;
    constexpr int kEpisodesPerEvaluation = 20;
    constexpr double kNoiseStdDev = 0.5;
    constexpr int kLogEvery = 25;
    double best_return =
        mujoco_rl_training::evaluate_average_return(env, best_policy, kEpisodesPerEvaluation, action_adapter);

    std::cout << "Initial Policy Return: " << best_return << std::endl;

    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, kNoiseStdDev);

    for (int itr = 0; itr < kNumIterations; ++itr) {
        mujoco_rl_training::DoublePendulumLinearPolicy positive_policy = best_policy;
        mujoco_rl_training::DoublePendulumLinearPolicy negative_policy = best_policy;

        for (std::size_t joint = 0; joint < positive_policy.bias.size(); ++joint) {
            const double bias_delta = noise(rng);
            positive_policy.bias[joint] += bias_delta;
            negative_policy.bias[joint] -= bias_delta;

            for (std::size_t i = 0; i < positive_policy.weights[joint].size(); ++i) {
                const double weight_delta = noise(rng);
                positive_policy.weights[joint][i] += weight_delta;
                negative_policy.weights[joint][i] -= weight_delta;
            }
        }

        const double positive_return =
            mujoco_rl_training::evaluate_average_return(env, positive_policy, kEpisodesPerEvaluation, action_adapter);
        const double negative_return =
            mujoco_rl_training::evaluate_average_return(env, negative_policy, kEpisodesPerEvaluation, action_adapter);

        if (positive_return > best_return || negative_return > best_return) {
            if (positive_return >= negative_return) {
                best_policy = positive_policy;
                best_return = positive_return;
            } else {
                best_policy = negative_policy;
                best_return = negative_return;
            }
        }

        if (itr % kLogEvery == 0) {
            std::cout << "iteration=" << itr << " positive_return=" << positive_return
                      << " negative_return=" << negative_return << " best_return=" << best_return << std::endl;
        }
    }

    std::cout << "Final best return: " << best_return << std::endl;
    mujoco_rl_training::save_double_pendulum_linear_policy(kPolicyArtifactPath, best_policy);
    const auto metadata_path = mujoco_rl_training::double_pendulum_metadata_path_for_policy(kPolicyArtifactPath);
    mujoco_rl_training::save_double_pendulum_policy_metadata(metadata_path, kPolicyArtifactPath, config, best_return,
                                                             kNumIterations, kEpisodesPerEvaluation, kNoiseStdDev);
    std::cout << "Saved best policy to: " << kPolicyArtifactPath << std::endl;
    std::cout << "Saved policy metadata to: " << metadata_path << std::endl;

    return 0;
}
