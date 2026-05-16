#include <mujoco_rl_training/PendulumLinearPolicy.h>
#include <envs/PendulumEnv.h>
#include <mujoco_rl_training/RolloutUtils.h>
#include <mujoco_rl_training/PendulumPolicyMetadata.h>
#include <mujoco_rl_training/PolicyIO.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cstddef>
#include <iostream>
#include <random>

namespace {

const char* kPolicyArtifactPath = "artifacts/pendulum_best_policy.txt";

void save_policy(const mujoco_rl_training::PendulumLinearPolicy& policy) {
    auto output = mujoco_rl_training::open_artifact_output(kPolicyArtifactPath);

    output << policy.weights[0] << ' ' << policy.weights[1] << ' ' << policy.weights[2] << ' ' << policy.bias << '\n';
}

const auto action_adapter = [](const auto& policy, const auto& observation, auto&) {
    return policy.action_from_obs(observation);
};

}  // namespace

int main() {
    mujoco_rl_training::PendulumEnvConfig config;
    config.episode_horizon = 400;
    config.max_torque = 40.0;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";

    mujoco_rl_training::PendulumEnv env(config);

    mujoco_rl_training::PendulumLinearPolicy best_policy{};
    constexpr int kNumIterations = 500;
    constexpr int kEpisodesPerEvaluation = 10;
    constexpr double kNoiseStddev = 1.0;
    constexpr int kLogEvery = 25;

    double best_return =
        mujoco_rl_training::evaluate_average_return(env, best_policy, kEpisodesPerEvaluation, action_adapter);

    std::cout << "Initial policy return: " << best_return << std::endl;

    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, kNoiseStddev);
    for (int itr = 0; itr < kNumIterations; ++itr) {
        mujoco_rl_training::PendulumLinearPolicy positive_policy = best_policy;
        mujoco_rl_training::PendulumLinearPolicy negative_policy = best_policy;

        const double bias_delta = noise(rng);
        positive_policy.bias += bias_delta;
        negative_policy.bias -= bias_delta;
        for (std::size_t i = 0; i < positive_policy.weights.size(); ++i) {
            const double weight_delta = noise(rng);
            positive_policy.weights[i] += weight_delta;
            negative_policy.weights[i] -= weight_delta;
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
    std::cout << "Best policy weights: [" << best_policy.weights[0] << ", " << best_policy.weights[1] << ", "
              << best_policy.weights[2] << "]" << std::endl;
    std::cout << "Best policy bias: " << best_policy.bias << std::endl;
    save_policy(best_policy);
    const auto metadata_path = mujoco_rl_training::pendulum_metadata_path_for_policy(kPolicyArtifactPath);
    mujoco_rl_training::save_pendulum_policy_metadata(metadata_path, kPolicyArtifactPath, config,
                                                      mujoco_rl_training::PendulumPolicyActionScale::PhysicalTorque,
                                                      "random_search", best_return);
    std::cout << "Saved best policy to: " << kPolicyArtifactPath << std::endl;
    std::cout << "Saved policy metadata to: " << metadata_path << std::endl;

    return 0;
}
