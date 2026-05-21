#pragma once

#include <envs/EnvTypes.hpp>
#include <mujoco_rl_training/ActionUtils.hpp>
#include <mujoco_rl_training/rl/TrajectoryUtils.h>
#include <mujoco_rl_training/torch/Actor.h>
#include <mujoco_rl_training/torch/Critic.h>
#include <mujoco_rl_training/torch/GaussianPolicy.h>
#include <mujoco_rl_training/torch/TensorUtils.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mujoco_rl_training {

struct PpoBatch {
    std::vector<std::vector<double>> observations;
    std::vector<std::vector<double>> raw_actions;
    std::vector<double> old_log_probs;
    std::vector<double> rewards;
    std::vector<double> values;
    std::vector<double> advantages;
    std::vector<double> returns;
    std::vector<bool> dones;
    double bootstrap_value = 0.0;

    void reserve(std::size_t size) {
        observations.reserve(size);
        raw_actions.reserve(size);
        old_log_probs.reserve(size);
        rewards.reserve(size);
        values.reserve(size);
        advantages.reserve(size);
        returns.reserve(size);
        dones.reserve(size);
    }

    void store(const std::vector<double>& observation, const std::vector<double>& raw_action, double old_log_prob,
               double reward, double value, bool done) {
        observations.push_back(observation);
        raw_actions.push_back(raw_action);
        old_log_probs.push_back(old_log_prob);
        rewards.push_back(reward);
        values.push_back(value);
        dones.push_back(done);
    }

    std::size_t size() const { return rewards.size(); }
};

inline double evaluate_critic_value(TorchCritic& critic, const std::vector<double>& observation,
                                    const torch::Device& device) {
    torch::NoGradGuard no_grad;
    const auto obs = vector_to_tensor(observation, device, true);
    return critic->forward(obs).item<double>();
}

template <typename Env, typename ResetFn>
PpoBatch collect_ppo_batch(Env& env, TorchActor& actor, TorchCritic& critic, const torch::Tensor& log_std,
                           const torch::Device& device, std::size_t steps_per_epoch, double reward_scale,
                           ResetFn reset_fn) {
    PpoBatch batch;
    batch.reserve(steps_per_epoch);
    auto observation = reset_fn(env);
    std::vector<double> last_next_observation = observation;
    bool last_done = false;

    actor->eval();
    critic->eval();

    while (batch.size() < steps_per_epoch) {
        torch::NoGradGuard no_grad;
        const auto observation_tensor = vector_to_tensor(observation, device, true);
        const auto mean = actor->forward(observation_tensor);
        const auto raw_action = sample_gaussian_action(mean, log_std).detach();
        const auto normalized_action = squash_action(raw_action);
        const torch::Tensor old_log_prob = squashed_gaussian_log_prob(raw_action, mean, log_std);
        const torch::Tensor value_tensor = critic->forward(observation_tensor);
        const auto value = value_tensor.item<double>();

        const auto action_command = scale_action_from_action_space(tensor_to_vector(normalized_action.squeeze(0)),
                                                                   env.env_spec().action_limit_space);
        const auto result = env.step(action_command);
        const auto done = result.terminated || result.truncated;

        batch.store(observation, tensor_to_vector(raw_action.squeeze(0)), old_log_prob.item<double>(),
                    reward_scale * result.reward, value, done);

        last_next_observation = result.observation;
        last_done = done;
        observation = result.observation;

        if (done) {
            observation = reset_fn(env);
        }
    }

    batch.bootstrap_value = last_done ? 0.0 : evaluate_critic_value(critic, last_next_observation, device);
    actor->train();
    critic->train();

    return batch;
}

struct EvaluationStats {
    double mean_return = 0.0;
};

template <typename Env, typename ResetFn>
EvaluationStats evaluate_mean_policy(Env& env, TorchActor& actor, const torch::Device& device, int episodes,
                                     ResetFn reset_fn) {
    torch::NoGradGuard no_grad;
    actor->eval();

    double total_return = 0.0;

    for (int episode = 0; episode < episodes; ++episode) {
        auto observation = reset_fn(env);
        double episode_return = 0.0;

        while (true) {
            const auto obs_tensor = vector_to_tensor(observation, device, true);
            const auto mean = actor->forward(obs_tensor);
            const auto normalized_action = squash_action(mean);
            const auto action_command = scale_action_from_action_space(
                tensor_to_vector(normalized_action.squeeze(0)), env.env_spec().action_limit_space);

            const auto result = env.step(action_command);

            episode_return += result.reward;
            observation = result.observation;

            if (result.terminated || result.truncated) {
                break;
            }
        }

        total_return += episode_return;
    }

    actor->train();

    EvaluationStats stats;
    stats.mean_return = total_return / static_cast<double>(episodes);
    return stats;
}

struct PpoUpdateStats {
    double actor_loss = 0.0;
    double critic_loss = 0.0;
    double approx_kl = 0.0;
    double clip_fraction = 0.0;
};

inline PpoUpdateStats update_ppo(TorchActor& actor, TorchCritic& critic, torch::optim::Adam& actor_optimizer,
                                 torch::optim::Adam& critic_optimizer, const torch::Tensor& log_std,
                                 const PpoBatch& batch, const torch::Device& device, int ppo_train_iters,
                                 int mini_batch_size, double clip_epsilon, double target_kl) {
    const auto observations = mujoco_rl_training::matrix_to_tensor(batch.observations, device);
    const auto raw_actions = mujoco_rl_training::matrix_to_tensor(batch.raw_actions, device);
    const auto old_log_probs = mujoco_rl_training::vector_to_column_tensor(batch.old_log_probs, device);
    const auto returns = mujoco_rl_training::vector_to_column_tensor(batch.returns, device);
    const auto advantages = mujoco_rl_training::vector_to_column_tensor(batch.advantages, device);

    PpoUpdateStats stats;

    const int64_t batch_size = observations.size(0);
    bool stop_early = false;
    int update_count = 0;
    double actor_loss_sum = 0.0;
    double critic_loss_sum = 0.0;
    double approx_kl_sum = 0.0;
    double clip_fraction_sum = 0.0;

    for (int iter = 0; iter < ppo_train_iters; ++iter) {
        const auto permutation = torch::randperm(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device));

        for (int64_t start = 0; start < batch_size; start += mini_batch_size) {
            const int64_t end = std::min(start + static_cast<int64_t>(mini_batch_size), batch_size);
            const auto indices = permutation.slice(0, start, end);

            const auto obs_mb = observations.index_select(0, indices);
            const auto raw_actions_mb = raw_actions.index_select(0, indices);
            const auto old_log_probs_mb = old_log_probs.index_select(0, indices);
            const auto returns_mb = returns.index_select(0, indices);
            const auto advantages_mb = advantages.index_select(0, indices);

            const auto mean = actor->forward(obs_mb);
            const auto log_probs = squashed_gaussian_log_prob(raw_actions_mb, mean, log_std);
            const auto ratio = torch::exp(log_probs - old_log_probs_mb);
            const auto unclipped = ratio * advantages_mb;
            const auto clipped = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb;
            const auto actor_loss = -torch::minimum(unclipped, clipped).mean();

            actor_optimizer.zero_grad();
            actor_loss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), 1.0);
            actor_optimizer.step();

            const auto values = critic->forward(obs_mb);
            const auto critic_loss = torch::mse_loss(values, returns_mb);

            critic_optimizer.zero_grad();
            critic_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), 1.0);
            critic_optimizer.step();

            const auto updated_mean = actor->forward(obs_mb);
            const auto updated_log_probs = squashed_gaussian_log_prob(raw_actions_mb, updated_mean, log_std);
            const auto updated_ratio = torch::exp(updated_log_probs - old_log_probs_mb);
            const double approx_kl = (old_log_probs_mb - updated_log_probs).mean().item<double>();
            const double clip_fraction =
                (torch::abs(updated_ratio - 1.0) > clip_epsilon).to(torch::kFloat32).mean().item<double>();

            actor_loss_sum += actor_loss.item<double>();
            critic_loss_sum += critic_loss.item<double>();
            approx_kl_sum += approx_kl;
            clip_fraction_sum += clip_fraction;
            ++update_count;

            if (approx_kl > target_kl) {
                stop_early = true;
                break;
            }
        }

        if (stop_early) {
            break;
        }
    }

    if (update_count > 0) {
        const double count = static_cast<double>(update_count);
        stats.actor_loss = actor_loss_sum / count;
        stats.critic_loss = critic_loss_sum / count;
        stats.approx_kl = approx_kl_sum / count;
        stats.clip_fraction = clip_fraction_sum / count;
    }

    return stats;
}

}  // namespace mujoco_rl_training
