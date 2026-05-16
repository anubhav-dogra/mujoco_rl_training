#include <envs/PendulumEnv.h>
#include <mujoco_rl_training/PendulumPolicyMetadata.h>
#include <mujoco_rl_training/RolloutUtils.h>
#include <mujoco_rl_training/artifacts/PolicyArtifacts.h>
#include <mujoco_rl_training/policies/PendulumGaussianPolicy.h>
#include <mujoco_rl_training/rl/TrajectoryUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

namespace {

const char* kPolicyArtifactPath = "artifacts/pendulum_vpg_policy.txt";

const auto mean_action_adapter = [](const auto& policy, const auto& observation, const auto& env) {
    return env.config().max_torque * policy.mean_action(observation);
};
}  // namespace

struct VpgBatch {
    std::vector<std::array<double, 3>> observations;
    std::vector<double> actions;
    std::vector<double> rewards;
    std::vector<double> values;
    std::vector<double> log_probs;
    std::vector<double> advantages;
    std::vector<double> returns;
    std::vector<bool> dones;
    double bootstrap_value = 0.0;

    void reserve(std::size_t size) {
        observations.reserve(size);
        actions.reserve(size);
        rewards.reserve(size);
        values.reserve(size);
        log_probs.reserve(size);
        advantages.reserve(size);
        returns.reserve(size);
        dones.reserve(size);
    }

    void store(const std::array<double, 3>& observation, double action, double reward, double value, double log_prob,
               bool done) {
        observations.push_back(observation);
        actions.push_back(action);
        rewards.push_back(reward);
        values.push_back(value);
        log_probs.push_back(log_prob);
        dones.push_back(done);
    }

    std::size_t size() const { return rewards.size(); }
};

VpgBatch collect_vpg_batch(mujoco_rl_training::PendulumEnv& env,
                           const mujoco_rl_training::PendulumGaussianPolicy& policy,
                           const mujoco_rl_training::PendulumValueFunction& critic, std::mt19937& rng,
                           std::size_t steps_per_epoch) {
    VpgBatch batch;
    batch.reserve(steps_per_epoch);
    auto observation = env.reset();

    while (batch.size() < steps_per_epoch) {
        const double normalized_action = policy.sample_action(observation, rng);
        const double action = env.config().max_torque * normalized_action;
        const double value = critic.predict(observation);
        const double log_prob = policy.log_probability(observation, normalized_action);

        const auto result = env.step(action);
        const bool done = result.terminated || result.truncated;
        batch.store(observation, normalized_action, result.reward, value, log_prob, done);
        observation = result.observation;
        if (done) {
            observation = env.reset();
        }
    }
    return batch;
}

void update_vpg_actor(mujoco_rl_training::PendulumGaussianPolicy& policy, const VpgBatch& batch, double learning_rate) {
    std::array<double, 3> grad_w = {0.0, 0.0, 0.0};
    double grad_b = 0.0;
    const double variance = policy.sigma * policy.sigma;

    for (std::size_t i = 0; i < batch.size(); ++i) {
        const double mu = policy.mean_action(batch.observations[i]);
        const double coeff = (batch.actions[i] - mu) / variance;
        const double advantage = batch.advantages[i];

        grad_w[0] += advantage * coeff * batch.observations[i][0];
        grad_w[1] += advantage * coeff * batch.observations[i][1];
        grad_w[2] += advantage * coeff * batch.observations[i][2];
        grad_b += advantage * coeff;
    }

    const double scale = 1.0 / static_cast<double>(batch.size());
    policy.weights[0] += learning_rate * grad_w[0] * scale;
    policy.weights[1] += learning_rate * grad_w[1] * scale;
    policy.weights[2] += learning_rate * grad_w[2] * scale;
    policy.bias += learning_rate * grad_b * scale;
}

void update_vpg_critic(mujoco_rl_training::PendulumValueFunction& critic, const VpgBatch& batch, double learning_rate,
                       int train_iters) {
    for (int iter = 0; iter < train_iters; ++iter) {
        std::array<double, 3> grad_w = {0.0, 0.0, 0.0};
        double grad_b = 0.0;

        for (std::size_t i = 0; i < batch.size(); ++i) {
            const double error = batch.returns[i] - critic.predict(batch.observations[i]);

            grad_w[0] += error * batch.observations[i][0];
            grad_w[1] += error * batch.observations[i][1];
            grad_w[2] += error * batch.observations[i][2];
            grad_b += error;
        }

        const double scale = 1.0 / static_cast<double>(batch.size());
        critic.weights[0] += learning_rate * grad_w[0] * scale;
        critic.weights[1] += learning_rate * grad_w[1] * scale;
        critic.weights[2] += learning_rate * grad_w[2] * scale;
        critic.bias += learning_rate * grad_b * scale;
    }
}

void vpg_update(mujoco_rl_training::PendulumGaussianPolicy& policy, mujoco_rl_training::PendulumValueFunction& critic,
                mujoco_rl_training::PendulumEnv& env, std::mt19937& rng, std::size_t steps_per_epoch, double gamma,
                double lambda, double learning_rate, double critic_learning_rate, int value_train_iters) {
    auto batch = collect_vpg_batch(env, policy, critic, rng, steps_per_epoch);
    mujoco_rl_training::compute_gae(batch, gamma, lambda);
    mujoco_rl_training::normalize_advantages(batch);
    update_vpg_actor(policy, batch, learning_rate);
    update_vpg_critic(critic, batch, critic_learning_rate, value_train_iters);
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
    mujoco_rl_training::PendulumValueFunction critic{};
    policy.sigma = 1.0;

    constexpr int kEpochs = 1000;
    constexpr std::size_t kStepsPerEpoch = 4000;
    constexpr double kGamma = 0.99;
    constexpr double kLambda = 0.95;
    constexpr double kActorLearningRate = 1e-1;
    constexpr double kCriticLearningRate = 1e-3;
    constexpr int kValueTrainIters = 40;
    constexpr int kLogEvery = 10;

    std::mt19937 rng(123);

    double best_mean_return = mujoco_rl_training::evaluate_episode_return(env, policy, mean_action_adapter);
    auto best_policy = policy;

    std::cout << "Initial mean-policy return: " << best_mean_return << std::endl;

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        vpg_update(policy, critic, env, rng, kStepsPerEpoch, kGamma, kLambda, kActorLearningRate, kCriticLearningRate,
                   kValueTrainIters);

        const double mean_return = mujoco_rl_training::evaluate_episode_return(env, policy, mean_action_adapter);
        if (mean_return > best_mean_return) {
            best_mean_return = mean_return;
            best_policy = policy;
        }

        if (epoch % kLogEvery == 0) {
            std::cout << "epoch=" << epoch << " mean_policy_return=" << mean_return
                      << " best_mean_return=" << best_mean_return << std::endl;
            /*std::cout << "weights=[" << policy.weights[0] << ", " << policy.weights[1] << ", " << policy.weights[2]*/
            /*          << "] bias=" << policy.bias << std::endl;*/
        }
    }

    policy = best_policy;
    mujoco_rl_training::save_pendulum_gaussian_policy(kPolicyArtifactPath, policy);
    const auto metadata_path = mujoco_rl_training::pendulum_metadata_path_for_policy(kPolicyArtifactPath);
    mujoco_rl_training::save_pendulum_policy_metadata(metadata_path, kPolicyArtifactPath, config,
                                                      mujoco_rl_training::PendulumPolicyActionScale::NormalizedTorque,
                                                      "vpg", best_mean_return);

    std::cout << "Final best mean-policy return: " << best_mean_return << std::endl;
    std::cout << "Best policy weights: [" << policy.weights[0] << ", " << policy.weights[1] << ", " << policy.weights[2]
              << "]" << std::endl;
    std::cout << "Best policy bias: " << policy.bias << std::endl;
    std::cout << "Best policy sigma: " << policy.sigma << std::endl;
    std::cout << "Saved best policy to: " << kPolicyArtifactPath << std::endl;
    std::cout << "Saved policy metadata to: " << metadata_path << std::endl;

    return 0;
}
