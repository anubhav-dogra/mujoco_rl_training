#pragma once

#include <envs/EnvTypes.hpp>
#include <mujoco_rl_training/ActionUtils.hpp>
#include <mujoco_rl_training/rl/RunningMeanStd.h>
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

// =============================================================================
// Proximal Policy Optimization (PPO) — single-actor, on-policy training loop.
//
// The high-level flow each epoch:
//
//   1. COLLECT:  Run the current policy in the env for kStepsPerEpoch steps,
//                storing (obs, action, reward, log_prob_old, value, done) for
//                each step. (`collect_ppo_batch`)
//   2. CREDIT:   Compute advantages with GAE and returns to regress the
//                critic against. (`compute_gae` + `normalize_advantages`)
//   3. UPDATE:   For several passes through the data, take SGD-style gradient
//                steps on the PPO clipped objective (actor) and MSE on
//                returns (critic). (`update_ppo`)
//   4. STOP IF:  Mean KL between new and old policy exceeds 1.5 * target_kl
//                — i.e. the policy moved too far in this update; cut the
//                remaining epochs to keep the on-policy assumption valid.
//
// PPO is "on-policy": data collected with the old policy can only be reused
// for a few SGD passes before it becomes stale. That's why we re-collect
// every epoch and only do `ppo_train_iters` passes through each batch.
// =============================================================================

// One epoch's worth of transitions, plus per-transition GAE outputs. The order
// of fields mirrors the order of operations: collected first, then GAE fills
// in advantages/returns from rewards + values.
struct PpoBatch {
    std::vector<std::vector<double>> observations;  // [T][obs_dim]   normalized
    std::vector<std::vector<double>> raw_actions;   // [T][action_dim] pre-tanh
    std::vector<double> old_log_probs;              // [T] log pi_old(a|s) — for PPO ratio
    std::vector<double> rewards;                    // [T] reward at step t (already scaled)
    std::vector<double> values;                     // [T] V(s_t) from critic at collection time
    std::vector<double> advantages;                 // [T] filled by compute_gae
    std::vector<double> returns;                    // [T] filled by compute_gae (= A + V)
    std::vector<bool> dones;                        // [T] episode boundary marker
    // V(s_{T+1}) at the post-rollout state (or 0 if the last step ended an
    // episode). Used by GAE to bootstrap past the end of the rollout.
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

// Single-observation critic forward, used to bootstrap V(s_{T+1}) at the end
// of a rollout. Wrapped in NoGradGuard since we don't need gradients here.
inline double evaluate_critic_value(TorchCritic& critic, const std::vector<double>& observation,
                                    const torch::Device& device) {
    torch::NoGradGuard no_grad;
    const auto obs = vector_to_tensor(observation, device, true);
    return critic->forward(obs).item<double>();
}

// -----------------------------------------------------------------------------
// collect_ppo_batch
//
// Runs the policy in the env for `steps_per_epoch` env steps, collecting
// everything PPO needs for one update. If an episode ends inside the budget,
// the env is reset via `reset_fn` and collection continues — the `dones[t]`
// flag marks the boundary so GAE doesn't credit reward across the reset.
//
// Observation normalization (optional, via `obs_normalizer`):
//   For each fresh observation we (a) fold it into the running mean/std,
//   (b) normalize it, (c) pass the *normalized* obs to actor/critic, and
//   (d) STORE the normalized obs in the batch. Step (d) is important: in
//   the update phase we re-forward through actor/critic with these stored
//   inputs, and they must match the distribution the policy was acting on.
//
// Both networks are put into eval() to disable any future BN/dropout
// behavior (none used today, but cheap insurance). The whole rollout runs
// under NoGradGuard since we only need values, not gradients.
// -----------------------------------------------------------------------------
template <typename Env, typename ResetFn>
PpoBatch collect_ppo_batch(Env& env, TorchActor& actor, TorchCritic& critic, const torch::Device& device,
                           std::size_t steps_per_epoch, double reward_scale, ResetFn reset_fn,
                           RunningMeanStd* obs_normalizer = nullptr) {
    PpoBatch batch;
    batch.reserve(steps_per_epoch);
    auto raw_observation = reset_fn(env);
    std::vector<double> last_raw_next_observation = raw_observation;
    bool last_done = false;

    // Local lambda so call sites don't repeat the null-check.
    auto normalize = [&](const std::vector<double>& obs) {
        return obs_normalizer ? obs_normalizer->normalize(obs) : obs;
    };

    actor->eval();
    critic->eval();

    while (batch.size() < steps_per_epoch) {
        // Fold in the new obs first, THEN normalize. Order is important: if we
        // didn't update, the very first obs would always look like an extreme
        // outlier (std=1, mean=0 at init), and we'd waste samples normalizing
        // against stale stats.
        if (obs_normalizer) {
            obs_normalizer->update(raw_observation);
        }
        const auto observation = normalize(raw_observation);

        torch::NoGradGuard no_grad;
        const auto observation_tensor = vector_to_tensor(observation, device, true);
        // Policy: forward -> mean, then sample raw_action ~ N(mean, exp(log_std))
        // and squash via tanh. Detach so we don't accidentally retain a graph.
        const auto mean = actor->forward(observation_tensor);
        const auto raw_action = sample_gaussian_action(mean, actor->log_std).detach();
        const auto normalized_action = squash_action(raw_action);
        // Snapshot log pi_old(a|s) at collection time — this is the ratio
        // denominator during the PPO update.
        const torch::Tensor old_log_prob = squashed_gaussian_log_prob(raw_action, mean, actor->log_std);
        // Critic baseline at the visited state.
        const torch::Tensor value_tensor = critic->forward(observation_tensor);
        const auto value = value_tensor.item<double>();

        // tanh-squashed action lives in (-1, 1); affine-map to the env's range.
        const auto action_command = scale_action_from_action_space(tensor_to_vector(normalized_action.squeeze(0)),
                                                                   env.env_spec().action_limit_space);
        const auto result = env.step(action_command);
        const auto done = result.terminated || result.truncated;

        // Save the NORMALIZED obs and raw (pre-tanh) action: PPO recomputes
        // log-probs in the update phase from these.
        batch.store(observation, tensor_to_vector(raw_action.squeeze(0)), old_log_prob.item<double>(),
                    reward_scale * result.reward, value, done);

        last_raw_next_observation = result.observation;
        last_done = done;

        // Reset on done to keep collecting; otherwise just advance.
        if (done) {
            raw_observation = reset_fn(env);
        } else {
            raw_observation = result.observation;
        }
    }

    // Bootstrap: V at the state AFTER the last action. If that step was a
    // terminal/truncated boundary, there's no future reward to bootstrap and
    // GAE uses 0 instead. (Note: we don't update the normalizer with this
    // tail observation — it's only used for one critic forward pass.)
    const auto last_next_observation = normalize(last_raw_next_observation);
    batch.bootstrap_value = last_done ? 0.0 : evaluate_critic_value(critic, last_next_observation, device);
    actor->train();
    critic->train();

    return batch;
}

// Output of `evaluate_mean_policy`. Only the mean episodic return for now —
// add per-scenario breakdown here if needed later.
struct EvaluationStats {
    double mean_return = 0.0;
};

// -----------------------------------------------------------------------------
// evaluate_mean_policy
//
// Greedy / deterministic evaluation: at each step we use the policy's MEAN
// action (no Gaussian noise), squash with tanh, and step the env. This
// removes exploration variance so the "best epoch" checkpoint signal isn't
// dominated by stochastic sampling.
//
// Important: this READS the running stats but does NOT update them. Training
// data should be what shapes the obs distribution, not eval rollouts.
// -----------------------------------------------------------------------------
template <typename Env, typename ResetFn>
EvaluationStats evaluate_mean_policy(Env& env, TorchActor& actor, const torch::Device& device, int episodes,
                                     ResetFn reset_fn, const RunningMeanStd* obs_normalizer = nullptr) {
    torch::NoGradGuard no_grad;
    actor->eval();

    double total_return = 0.0;

    auto normalize = [&](const std::vector<double>& obs) {
        return obs_normalizer ? obs_normalizer->normalize(obs) : obs;
    };

    for (int episode = 0; episode < episodes; ++episode) {
        auto raw_observation = reset_fn(env);
        double episode_return = 0.0;

        while (true) {
            // Deterministic action: a = tanh(mu(s)). No sampling.
            const auto obs_tensor = vector_to_tensor(normalize(raw_observation), device, true);
            const auto mean = actor->forward(obs_tensor);
            const auto normalized_action = squash_action(mean);
            const auto action_command = scale_action_from_action_space(
                tensor_to_vector(normalized_action.squeeze(0)), env.env_spec().action_limit_space);

            const auto result = env.step(action_command);

            episode_return += result.reward;
            raw_observation = result.observation;

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

// Diagnostic numbers reported by update_ppo each epoch. Useful for spotting
// pathologies — clip_fraction stuck near 0 means clipping isn't biting and
// you can loosen the gate; near 1 means you're clipping everything and the
// learning rate / clip range is wrong.
struct PpoUpdateStats {
    double actor_loss = 0.0;     // mean negative clipped objective (lower = better)
    double critic_loss = 0.0;    // mean MSE between critic and returns
    double approx_kl = 0.0;      // Schulman k3 estimate of KL(pi_old || pi_new)
    double clip_fraction = 0.0;  // fraction of samples where the ratio was outside [1-eps, 1+eps]
    double entropy = 0.0;        // analytic raw-Gaussian entropy at the end
    double mean_std = 0.0;       // mean exp(log_std) across action dims (exploration noise scale)
};

// -----------------------------------------------------------------------------
// update_ppo
//
// Does up to `ppo_train_iters` passes over the batch, taking a gradient step
// per mini-batch on:
//
//   Actor loss (CLIPPED PPO OBJECTIVE):
//
//     ratio_t = exp( log pi_new(a_t|s_t) - log pi_old(a_t|s_t) )
//     L_t     = min( ratio_t * A_t,
//                    clip(ratio_t, 1-eps, 1+eps) * A_t )
//     loss    = -mean(L_t) - entropy_coef * H(pi)
//
//   The `min` is the key PPO trick. If the policy improves the action's
//   advantage too aggressively (ratio shoots up), the clipped term takes over
//   and removes the incentive to push further. If the policy makes things
//   worse, the unclipped term ensures we still get the corrective gradient.
//
//   Critic loss: MSE between V(s) and the GAE returns target.
//
// Early stopping (per-epoch, not per-mini-batch):
//   After each full pass, we compute mean KL across the WHOLE batch and
//   abort the remaining epochs if it exceeds 1.5 * target_kl. KL is estimated
//   with the Schulman k3 formula: (r - 1) - log(r). This estimator is
//   low-variance and non-negative, much friendlier to threshold than the raw
//   (log_old - log_new).mean() (k1) form.
//
// log_std update:
//   `log_std` is a parameter of `actor`, so `actor_optimizer.step()` updates
//   it alongside the network weights. After each step we clamp it to a sane
//   range to prevent collapse (huge negative log_std → deterministic policy →
//   stuck) or explosion.
// -----------------------------------------------------------------------------
inline PpoUpdateStats update_ppo(TorchActor& actor, TorchCritic& critic, torch::optim::Adam& actor_optimizer,
                                 torch::optim::Adam& critic_optimizer, const PpoBatch& batch,
                                 const torch::Device& device, int ppo_train_iters, int mini_batch_size,
                                 double clip_epsilon, double target_kl, double entropy_coef = 0.0,
                                 double log_std_min = -5.0, double log_std_max = 2.0) {
    // Move the whole batch onto the device once — far cheaper than per-step.
    const auto observations = mujoco_rl_training::matrix_to_tensor(batch.observations, device);
    const auto raw_actions = mujoco_rl_training::matrix_to_tensor(batch.raw_actions, device);
    const auto old_log_probs = mujoco_rl_training::vector_to_column_tensor(batch.old_log_probs, device);
    const auto returns = mujoco_rl_training::vector_to_column_tensor(batch.returns, device);
    const auto advantages = mujoco_rl_training::vector_to_column_tensor(batch.advantages, device);

    PpoUpdateStats stats;

    const int64_t batch_size = observations.size(0);
    double actor_loss_sum = 0.0;
    double critic_loss_sum = 0.0;
    double clip_fraction_sum = 0.0;
    int update_count = 0;
    double final_epoch_kl = 0.0;

    // -- PPO epochs over the batch ------------------------------------------
    for (int iter = 0; iter < ppo_train_iters; ++iter) {
        // Shuffle indices each epoch so mini-batch boundaries don't always
        // line up the same way (standard SGD practice).
        const auto permutation = torch::randperm(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device));

        // -- mini-batches -------------------------------------------------
        for (int64_t start = 0; start < batch_size; start += mini_batch_size) {
            const int64_t end = std::min(start + static_cast<int64_t>(mini_batch_size), batch_size);
            const auto indices = permutation.slice(0, start, end);

            // Gather this mini-batch's slice of every tensor.
            const auto obs_mb = observations.index_select(0, indices);
            const auto raw_actions_mb = raw_actions.index_select(0, indices);
            const auto old_log_probs_mb = old_log_probs.index_select(0, indices);
            const auto returns_mb = returns.index_select(0, indices);
            const auto advantages_mb = advantages.index_select(0, indices);

            // Forward the CURRENT policy on the SAME (s, a) pairs we collected
            // under the OLD policy. The ratio compares the two.
            const auto mean = actor->forward(obs_mb);
            const auto log_probs = squashed_gaussian_log_prob(raw_actions_mb, mean, actor->log_std);
            const auto log_ratio = log_probs - old_log_probs_mb;
            const auto ratio = torch::exp(log_ratio);

            // Two-sided clipped surrogate objective (see header docstring).
            const auto unclipped = ratio * advantages_mb;
            const auto clipped = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_mb;

            // Entropy bonus, analytic for raw (pre-tanh) Gaussian:
            //   H(N(mu, sigma)) = sum_d [0.5 * log(2*pi*e) + log_std_d]
            // We add it as `-entropy_coef * H` (subtract entropy from loss to
            // ENCOURAGE entropy). Acts as a regularizer against premature
            // collapse to a deterministic policy.
            constexpr double kHalfLogTwoPiE = 1.4189385332046727;
            const auto entropy = (actor->log_std + kHalfLogTwoPiE).sum();
            const auto actor_loss = -torch::minimum(unclipped, clipped).mean() - entropy_coef * entropy;

            actor_optimizer.zero_grad();
            actor_loss.backward();
            // Global norm clipping — keep any one update from being huge.
            torch::nn::utils::clip_grad_norm_(actor->parameters(), 1.0);
            actor_optimizer.step();

            // Keep log_std in a sane range. Without this, a few bad updates
            // can drag it to -inf (policy becomes deterministic and can't
            // explore out) or +inf (action sampling becomes garbage noise).
            {
                torch::NoGradGuard no_grad;
                actor->log_std.clamp_(log_std_min, log_std_max);
            }

            // Critic step is independent — different parameters, different optimizer.
            const auto values = critic->forward(obs_mb);
            const auto critic_loss = torch::mse_loss(values, returns_mb);

            critic_optimizer.zero_grad();
            critic_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), 1.0);
            critic_optimizer.step();

            // Diagnostic: fraction of samples that hit the clip region.
            // Computed on the pre-update ratio because we want to characterize
            // the gradient signal we just consumed.
            double clip_fraction;
            {
                torch::NoGradGuard no_grad;
                clip_fraction = (torch::abs(ratio - 1.0) > clip_epsilon).to(torch::kFloat32).mean().item<double>();
            }

            actor_loss_sum += actor_loss.item<double>();
            critic_loss_sum += critic_loss.item<double>();
            clip_fraction_sum += clip_fraction;
            ++update_count;
        }

        // -- KL check on the WHOLE batch (not just last mini-batch) -------
        // Spinning Up's recipe: after each epoch, measure how far the policy
        // moved overall and abort the remaining epochs if it moved too far.
        // Per-mini-batch checks abort too early (noise) and waste data.
        {
            torch::NoGradGuard no_grad;
            const auto mean = actor->forward(observations);
            const auto log_probs = squashed_gaussian_log_prob(raw_actions, mean, actor->log_std);
            const auto log_ratio = log_probs - old_log_probs;
            const auto ratio = torch::exp(log_ratio);
            // Schulman k3 KL estimator: (r - 1) - log(r). Non-negative,
            // low-variance, and exact in expectation for KL(old || new).
            final_epoch_kl = (ratio - 1.0 - log_ratio).mean().item<double>();
        }

        if (final_epoch_kl > 1.5 * target_kl) {
            break;
        }
    }

    // Average the accumulated diagnostics over actual mini-batches taken.
    if (update_count > 0) {
        const double count = static_cast<double>(update_count);
        stats.actor_loss = actor_loss_sum / count;
        stats.critic_loss = critic_loss_sum / count;
        stats.clip_fraction = clip_fraction_sum / count;
    }
    stats.approx_kl = final_epoch_kl;
    {
        torch::NoGradGuard no_grad;
        constexpr double kHalfLogTwoPiE = 1.4189385332046727;
        stats.entropy = (actor->log_std + kHalfLogTwoPiE).sum().item<double>();
        stats.mean_std = torch::exp(actor->log_std).mean().item<double>();
    }

    return stats;
}

}  // namespace mujoco_rl_training
