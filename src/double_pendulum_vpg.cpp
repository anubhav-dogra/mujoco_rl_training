#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_rl_training/DoublePendulumGaussianPolicy.hpp>
#include <mujoco_rl_training/DoublePendulumLinearPolicy.h>
#include <mujoco_rl_training/DoublePendulumPolicyMetadata.h>
#include <mujoco_rl_training/PolicyIO.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kPolicyArtifactPath = "artifacts/double_pendulum_vpg_policy.txt";

std::vector<double> scale_action(const std::vector<double>& normalized_action, const std::vector<double>& max_torques) {
    if (normalized_action.size() != max_torques.size()) {
        throw std::runtime_error("Normalized action size must match max_torques size");
    }

    std::vector<double> action(normalized_action.size(), 0.0);
    for (std::size_t i = 0; i < normalized_action.size(); ++i) {
        action[i] = normalized_action[i] * max_torques[i];
    }
    return action;
}

mujoco_rl_training::DoublePendulumLinearPolicy to_physical_mean_policy(
    const mujoco_rl_training::DoublePendulumGaussianPolicy& policy, const std::vector<double>& max_torques) {
    if (max_torques.size() != mujoco_rl_training::DoublePendulumGaussianPolicy::kActionDim) {
        throw std::runtime_error("Double pendulum VPG expected exactly 2 max torques");
    }

    mujoco_rl_training::DoublePendulumLinearPolicy physical_policy;
    for (std::size_t action_index = 0; action_index < policy.weights.size(); ++action_index) {
        for (std::size_t obs_index = 0; obs_index < policy.weights[action_index].size(); ++obs_index) {
            physical_policy.weights[action_index][obs_index] =
                max_torques[action_index] * policy.weights[action_index][obs_index];
        }
        physical_policy.bias[action_index] = max_torques[action_index] * policy.bias[action_index];
    }
    return physical_policy;
}

void save_policy(const mujoco_rl_training::DoublePendulumGaussianPolicy& policy,
                 const mujoco_rl_training::DoublePendulumEnvConfig& config) {
    const auto physical_policy = to_physical_mean_policy(policy, config.max_torques);
    auto output = mujoco_rl_training::open_artifact_output(kPolicyArtifactPath);
    output << std::setprecision(17);

    for (const auto& row : physical_policy.weights) {
        for (double weight : row) {
            output << weight << ' ';
        }
    }
    output << physical_policy.bias[0] << ' ' << physical_policy.bias[1] << '\n';
}

double evaluate_mean_policy(mujoco_rl_training::DoublePendulumEnv& env,
                            const mujoco_rl_training::DoublePendulumGaussianPolicy& policy, int episodes) {
    double total_return = 0.0;

    for (int episode = 0; episode < episodes; ++episode) {
        auto observation = env.reset();
        double episode_return = 0.0;

        while (true) {
            const auto normalized_action = policy.mean_action(observation);
            const auto action = scale_action(normalized_action, env.config().max_torques);
            const auto result = env.step(action);
            episode_return += result.reward;
            observation = result.observation;

            if (result.truncated || result.terminated) {
                break;
            }
        }

        total_return += episode_return;
    }

    return total_return / static_cast<double>(episodes);
}

}  // namespace

// struct for a VPGbatch, collecting everything for the full epoch run.
struct VpgBatch {
    std::vector<std::vector<double>> observations;
    std::vector<std::vector<double>> actions;
    std::vector<double> rewards;
    std::vector<double> values;
    std::vector<double> log_probabilities;
    std::vector<double> advantages;
    std::vector<double> returns;
    std::vector<bool> dones;

    void reserve(std::size_t size) {
        observations.reserve(size);
        actions.reserve(size);
        rewards.reserve(size);
        values.reserve(size);
        log_probabilities.reserve(size);
        advantages.reserve(size);
        returns.reserve(size);
        dones.reserve(size);
    }
    void store(const std::vector<double>& observation, const std::vector<double>& action, double reward, double value,
               double log_prob, bool done) {
        observations.push_back(observation);
        actions.push_back(action);
        rewards.push_back(reward);
        values.push_back(value);
        log_probabilities.push_back(log_prob);
        dones.push_back(done);
    }

    std::size_t size() const { return rewards.size(); }
};

// function for collecting the batch when running epoch.
VpgBatch collect_vpg_batch(mujoco_rl_training::DoublePendulumEnv& env,
                           const mujoco_rl_training::DoublePendulumGaussianPolicy& policy,
                           const mujoco_rl_training::DoublePendulumValueFunction& critic, std::size_t steps_per_epochs,
                           std::mt19937& rng) {
    VpgBatch batch;
    batch.reserve(steps_per_epochs);
    auto obs = env.reset();

    while (batch.size() < steps_per_epochs) {
        const auto normalized_action = policy.sample_action(obs, rng);
        const auto action = scale_action(normalized_action, env.config().max_torques);
        const auto value = critic.predict(obs);
        const auto log_prob = policy.log_probability(obs, normalized_action);

        const auto result = env.step(action);
        const bool done = result.terminated || result.truncated;
        batch.store(obs, normalized_action, result.reward, value, log_prob, done);
        obs = result.observation;
        if (done) {
            obs = env.reset();
        }
    }
    return batch;
}

void compute_gae(VpgBatch& batch, double gamma, double lambda) {
    batch.advantages.assign(batch.size(), 0.0);
    batch.returns.assign(batch.size(), 0.0);

    if (batch.size() == 0) {
        return;
    }
    double gae = 0.0;
    double next_value = 0.0;
    for (int t = static_cast<int>(batch.size()) - 1; t >= 0; --t) {
        const std::size_t index = static_cast<std::size_t>(t);
        const double mask = batch.dones[index] ? 0.0 : 1.0;
        // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        const double delta = batch.rewards[index] + (gamma * next_value * mask) - batch.values[index];
        /*advantage_t = delta_t + gamma * lambda * delta_{t+1}*/
        /*              + (gamma * lambda)^2 * delta_{t+2}*/
        /*              + ...*/
        gae = delta + gamma * lambda * mask * gae;
        batch.advantages[index] = gae;
        batch.returns[index] = gae + batch.values[index];

        next_value = batch.values[index];
    }
}
void normalize_advantages(VpgBatch& batch) {
    if (batch.advantages.empty()) {
        return;
    }

    double sum = 0.0;
    double sq_sum = 0.0;

    for (double advantage : batch.advantages) {
        sum += advantage;
        sq_sum += advantage * advantage;
    }

    const double count = static_cast<double>(batch.advantages.size());
    const double mean = sum / count;
    const double variance = (sq_sum / count) - (mean * mean);
    const double stddev = std::sqrt(std::max(variance, 1e-8));

    for (double& advantage : batch.advantages) {
        advantage = (advantage - mean) / stddev;
    }
}

void update_vpg_actor(mujoco_rl_training::DoublePendulumGaussianPolicy& policy, const VpgBatch& batch,
                      double learning_rate) {
    std::vector<std::vector<double>> grad_w(policy.weights.size(), std::vector<double>(policy.weights[0].size(), 0.0));
    std::vector<double> grad_b(policy.bias.size(), 0.0);

    const double variance = policy.sigma * policy.sigma;

    for (std::size_t t = 0; t < batch.size(); ++t) {
        const auto mu = policy.mean_action(batch.observations[t]);
        for (std::size_t action_index = 0; action_index < policy.bias.size(); ++action_index) {
            const auto coeff = (batch.actions[t][action_index] - mu[action_index]) / variance;
            const double advantage = batch.advantages[t];
            for (std::size_t j = 0; j < batch.observations[t].size(); ++j) {
                grad_w[action_index][j] += advantage * coeff * batch.observations[t][j];
            }
            grad_b[action_index] += advantage * coeff;
        }
    }
    const double scale = 1.0 / static_cast<double>(batch.size());

    for (std::size_t a = 0; a < policy.bias.size(); ++a) {
        for (std::size_t j = 0; j < policy.weights[a].size(); ++j) {
            policy.weights[a][j] += learning_rate * grad_w[a][j] * scale;
        }

        policy.bias[a] += learning_rate * grad_b[a] * scale;
    }
}
void update_vpg_critic(mujoco_rl_training::DoublePendulumValueFunction& critic, const VpgBatch& batch,
                       double learning_rate, int train_iters) {
    for (int iter = 0; iter < train_iters; ++iter) {
        std::vector<double> grad_w(critic.weights.size(), 0.0);
        double grad_b = 0.0;

        for (std::size_t t = 0; t < batch.size(); ++t) {
            const double error = batch.returns[t] - critic.predict(batch.observations[t]);

            for (std::size_t j = 0; j < batch.observations[t].size(); ++j) {
                grad_w[j] += error * batch.observations[t][j];
            }

            grad_b += error;
        }

        const double scale = 1.0 / static_cast<double>(batch.size());

        for (std::size_t j = 0; j < critic.weights.size(); ++j) {
            critic.weights[j] += learning_rate * grad_w[j] * scale;
        }

        critic.bias += learning_rate * grad_b * scale;
    }
}
void vpg_update(mujoco_rl_training::DoublePendulumGaussianPolicy& policy,
                mujoco_rl_training::DoublePendulumValueFunction& critic, mujoco_rl_training::DoublePendulumEnv& env,
                std::mt19937& rng, std::size_t steps_per_epoch, double gamma, double lambda, double actor_learning_rate,
                double critic_learning_rate, int value_train_iters) {
    auto batch = collect_vpg_batch(env, policy, critic, steps_per_epoch, rng);
    compute_gae(batch, gamma, lambda);
    normalize_advantages(batch);
    update_vpg_actor(policy, batch, actor_learning_rate);
    update_vpg_critic(critic, batch, critic_learning_rate, value_train_iters);
}

int main() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {1.0, 1.0};
    config.velocity_cost_weights = {0.01, 0.01};
    config.control_cost_weights = {0.001, 0.001};
    config.max_torques = {40, 30};
    config.episode_horizon = 400;
    config.repeat_action = 20;
    config.simulation_frequency = 1000;

    mujoco_rl_training::DoublePendulumEnv env(config);

    mujoco_rl_training::DoublePendulumGaussianPolicy policy{};
    mujoco_rl_training::DoublePendulumValueFunction critic{};
    policy.sigma = 1.0;

    constexpr int kEpochs = 1000;
    constexpr std::size_t kStepsPerEpoch = 4000;
    constexpr int kEpisodesPerEvaluation = 1;
    constexpr double kGamma = 0.99;
    constexpr double kLambda = 0.95;
    constexpr double kActorLearningRate = 0.1;
    constexpr double kCriticLearningRate = 0.001;
    constexpr int kValueTrainIters = 40;
    constexpr int kLogEvery = 10;

    std::mt19937 rng(123);

    double best_mean_return = evaluate_mean_policy(env, policy, kEpisodesPerEvaluation);
    auto best_policy = policy;

    std::cout << "Initial mean-policy return: " << best_mean_return << std::endl;

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        vpg_update(policy, critic, env, rng, kStepsPerEpoch, kGamma, kLambda, kActorLearningRate, kCriticLearningRate,
                   kValueTrainIters);

        const double mean_return = evaluate_mean_policy(env, policy, kEpisodesPerEvaluation);
        if (mean_return > best_mean_return) {
            best_mean_return = mean_return;
            best_policy = policy;
        }

        if (epoch % kLogEvery == 0) {
            std::cout << "epoch=" << epoch << " mean_policy_return=" << mean_return
                      << " best_mean_return=" << best_mean_return << std::endl;
        }
    }

    policy = best_policy;
    save_policy(policy, config);
    const auto metadata_path = mujoco_rl_training::double_pendulum_metadata_path_for_policy(kPolicyArtifactPath);
    mujoco_rl_training::save_double_pendulum_policy_metadata(
        metadata_path, kPolicyArtifactPath, config, best_mean_return, kEpochs, kEpisodesPerEvaluation, policy.sigma);

    std::cout << "Final best mean-policy return: " << best_mean_return << std::endl;
    std::cout << "Best policy bias: [" << policy.bias[0] << ", " << policy.bias[1] << "]" << std::endl;
    std::cout << "Best policy sigma: " << policy.sigma << std::endl;
    std::cout << "Saved best mean policy to: " << kPolicyArtifactPath << std::endl;
    std::cout << "Saved policy metadata to: " << metadata_path << std::endl;

    return 0;
}
