#include <mujoco_rl_training/PendulumLinearPolicy.h>
#include <mujoco_rl_training/PendulumEnv.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

const char* kPolicyArtifactPath = "artifacts/pendulum_best_policy.txt";

void save_policy(const mujoco_rl_training::PendulumLinearPolicy& policy) {
    std::filesystem::create_directories("artifacts");

    std::ofstream output(kPolicyArtifactPath, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open policy artifact for writing.");
    }

    output << policy.weights[0] << ' '
           << policy.weights[1] << ' '
           << policy.weights[2] << ' '
           << policy.bias << '\n';
}

}  // namespace

double evaluate_episode_return(mujoco_rl_training::PendulumEnv& env,
                               const mujoco_rl_training::PendulumLinearPolicy& policy) {
    auto obs = env.reset();
    double total_reward = 0.0;

    while (true) {
        const double action = policy.action_from_obs(obs);
        const auto result = env.step(action);
        obs = result.observation;

        total_reward += result.reward;
        if (result.truncated || result.terminated) {
            break;
        }
    }
    return total_reward;
}

double evaluate_average_return(mujoco_rl_training::PendulumEnv& env,
                               const mujoco_rl_training::PendulumLinearPolicy& policy, int num_episodes) {
    double total_return = 0.0;
    for (int episode = 0; episode < num_episodes; ++episode) {
        total_return += evaluate_episode_return(env, policy);
    }
    return total_return / static_cast<double>(num_episodes);
}

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

    double best_return = evaluate_average_return(env, best_policy, kEpisodesPerEvaluation);

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

        const double positive_return = evaluate_average_return(env, positive_policy, kEpisodesPerEvaluation);
        const double negative_return = evaluate_average_return(env, negative_policy, kEpisodesPerEvaluation);

        if (positive_return > best_return || negative_return > best_return) {
            if (positive_return >= negative_return) {
                best_policy = positive_policy;
                best_return = positive_return;
            } else {
                best_policy = negative_policy;
                best_return = negative_return;
            }
        }
    }

    std::cout << "Final best return: " << best_return << std::endl;
    std::cout << "Best policy weights: [" << best_policy.weights[0] << ", " << best_policy.weights[1] << ", "
              << best_policy.weights[2] << "]" << std::endl;
    std::cout << "Best policy bias: " << best_policy.bias << std::endl;
    save_policy(best_policy);
    std::cout << "Saved best policy to: " << kPolicyArtifactPath << std::endl;

    return 0;
}
