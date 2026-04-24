#include <mujoco_rl_training/PendulumGaussianPolicy.h>
#include <mujoco_rl_training/PendulumEnv.h>
#include <mujoco_rl_training/PolicyIO.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

const char* kPolicyArtifactPath = "artifacts/pendulum_reinforce_policy.txt";

void save_policy(const mujoco_rl_training::PendulumGaussianPolicy& policy) {
    auto output = mujoco_rl_training::open_artifact_output(kPolicyArtifactPath);

    output << policy.weights[0] << ' ' << policy.weights[1] << ' ' << policy.weights[2] << ' ' << policy.bias << ' '
           << policy.sigma << '\n';
}

double evaluate_mean_policy(mujoco_rl_training::PendulumEnv& env,
                            const mujoco_rl_training::PendulumGaussianPolicy& policy) {
    auto observation = env.reset();
    double total_return = 0.0;

    while (true) {
        const double action = policy.mean_action(observation);
        const auto result = env.step(action);
        total_return += result.reward;
        observation = result.observation;

        if (result.truncated || result.terminated) {
            break;
        }
    }

    return total_return;
}

}  // namespace

mujoco_rl_training::Trajectory collect_trajectory(mujoco_rl_training::PendulumEnv& env,
                                                  const mujoco_rl_training::PendulumGaussianPolicy& policy,
                                                  std::mt19937& rng) {
    /* For rollout data, store the tuple:

   (s_t, a_t, logpi_t, r_t)*/
    auto s_t = env.reset();
    mujoco_rl_training::Trajectory traj;
    while (true) {
        const double a_t = policy.sample_action(s_t, rng);
        const double logpi_t = policy.log_probability(s_t, a_t);

        mujoco_rl_training::TrajectoryStep ts;
        ts.log_probability = logpi_t;
        ts.observation = s_t;  // observation which caused the action (in REINFORCE)
        ts.sampled_action = a_t;

        const auto result = env.step(a_t);
        ts.reward = result.reward;
        traj.steps.push_back(ts);

        s_t = result.observation;

        if (result.truncated || result.terminated) {
            break;
        }
    }
    return traj;
}
/*G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...*/
std::vector<double> compute_discounted_returns(const mujoco_rl_training::Trajectory& trajectory, double gamma) {
    std::vector<double> discounted_returns(trajectory.steps.size(), 0.0);
    double running_return = 0.0;

    for (int t = static_cast<int>(trajectory.steps.size()) - 1; t >= 0; --t) {
        running_return = trajectory.steps[static_cast<std::size_t>(t)].reward + gamma * running_return;
        discounted_returns[static_cast<std::size_t>(t)] = running_return;
    }

    return discounted_returns;
}

// Collect a small on-policy batch, center returns with a batch baseline, and
// apply one manual REINFORCE update for the fixed-variance Gaussian policy.
void reinforce_update_batch(mujoco_rl_training::PendulumGaussianPolicy& policy, mujoco_rl_training::PendulumEnv& env,
                            std::mt19937& rng, double gamma, double learning_rate, int episodes_per_update) {
    std::array<double, 3> grad_w = {0.0, 0.0, 0.0};
    double grad_b = 0.0;
    const double variance = policy.sigma * policy.sigma;
    std::vector<mujoco_rl_training::Trajectory> trajectories;
    std::vector<std::vector<double>> discounted_return_batches;
    trajectories.reserve(static_cast<std::size_t>(episodes_per_update));
    discounted_return_batches.reserve(static_cast<std::size_t>(episodes_per_update));

    double batch_return_sum = 0.0;
    std::size_t batch_return_count = 0;

    for (int episode = 0; episode < episodes_per_update; ++episode) {
        const auto trajectory = collect_trajectory(env, policy, rng);
        const auto discounted_returns = compute_discounted_returns(trajectory, gamma);
        for (double value : discounted_returns) {
            batch_return_sum += value;
            ++batch_return_count;
        }

        trajectories.push_back(trajectory);
        discounted_return_batches.push_back(discounted_returns);
    }

    const double baseline = batch_return_sum / static_cast<double>(batch_return_count);

    for (std::size_t episode = 0; episode < trajectories.size(); ++episode) {
        const auto& trajectory = trajectories[episode];
        const auto& discounted_returns = discounted_return_batches[episode];

        for (int t = 0; t < static_cast<int>(trajectory.steps.size()); ++t) {
            const double mu_t = policy.mean_action(trajectory.steps[t].observation);
            const double coeff = (trajectory.steps[t].sampled_action - mu_t) / variance;
            const double advantage = discounted_returns[static_cast<std::size_t>(t)] - baseline;
            grad_w[0] += advantage * coeff * trajectory.steps[t].observation[0];
            grad_w[1] += advantage * coeff * trajectory.steps[t].observation[1];
            grad_w[2] += advantage * coeff * trajectory.steps[t].observation[2];
            grad_b += advantage * coeff;
        }
    }

    const double batch_scale = 1.0 / static_cast<double>(episodes_per_update);
    policy.weights[0] += learning_rate * grad_w[0] * batch_scale;
    policy.weights[1] += learning_rate * grad_w[1] * batch_scale;
    policy.weights[2] += learning_rate * grad_w[2] * batch_scale;
    policy.bias += learning_rate * grad_b * batch_scale;
}

int main() {
    mujoco_rl_training::PendulumEnvConfig config;
    config.episode_horizon = 400;
    config.max_torque = 20.0;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";

    mujoco_rl_training::PendulumEnv env(config);
    mujoco_rl_training::PendulumGaussianPolicy policy{};
    policy.sigma = 0.1;

    constexpr int kNumIterations = 500;
    constexpr double kGamma = 0.99;
    constexpr double kLearningRate = 1e-4;
    constexpr int kLogEvery = 25;
    constexpr int kEpisodesPerUpdate = 5;

    std::mt19937 rng(123);

    double best_mean_return = evaluate_mean_policy(env, policy);
    mujoco_rl_training::PendulumGaussianPolicy best_policy = policy;

    std::cout << "Initial mean-policy return: " << best_mean_return << std::endl;

    for (int iteration = 0; iteration < kNumIterations; ++iteration) {
        reinforce_update_batch(policy, env, rng, kGamma, kLearningRate, kEpisodesPerUpdate);

        const double mean_return = evaluate_mean_policy(env, policy);
        if (mean_return > best_mean_return) {
            best_mean_return = mean_return;
            best_policy = policy;
        }

        if (iteration % kLogEvery == 0) {
            std::cout << "iteration=" << iteration << " episodes_per_update=" << kEpisodesPerUpdate
                      << " mean_policy_return=" << mean_return << " best_mean_return=" << best_mean_return << std::endl;
        }
    }

    policy = best_policy;
    save_policy(policy);
    std::cout << "Final best mean-policy return: " << best_mean_return << std::endl;
    std::cout << "Best policy weights: [" << policy.weights[0] << ", " << policy.weights[1] << ", " << policy.weights[2]
              << "]" << std::endl;
    std::cout << "Best policy bias: " << policy.bias << std::endl;
    std::cout << "Best policy sigma: " << policy.sigma << std::endl;
    std::cout << "Saved best policy to: " << kPolicyArtifactPath << std::endl;

    return 0;
}
