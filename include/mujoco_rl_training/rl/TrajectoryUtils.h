#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mujoco_rl_training {

// =============================================================================
// Generalized Advantage Estimation (GAE) and advantage normalization.
//
// Background. The "advantage" A(s_t, a_t) = Q(s_t, a_t) - V(s_t) measures how
// much better the action taken was than the value function expected. Policy
// gradient methods (including PPO) work better when each gradient sample is
// scaled by the advantage rather than the raw return, because the V(s)
// baseline reduces variance without introducing bias.
//
// Plain Monte Carlo advantage is unbiased but high-variance:
//     A_MC = sum_{k>=0} gamma^k * r_{t+k} - V(s_t)
//
// Plain TD advantage is low-variance but biased:
//     A_TD = r_t + gamma * V(s_{t+1}) - V(s_t) = delta_t
//
// GAE blends them: with parameter lambda in [0, 1],
//     A_GAE(lambda) = sum_{k>=0} (gamma*lambda)^k * delta_{t+k}
//
// lambda = 0 reduces to TD (biased, low-variance).
// lambda = 1 reduces to MC (unbiased, high-variance).
// lambda ~ 0.95 is the standard PPO choice — empirically a good trade-off.
//
// The recurrence we actually use, working backwards from the last step:
//     gae_t = delta_t + gamma * lambda * gae_{t+1}
// where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
// =============================================================================

template <typename Trajectory>
inline void compute_gae(Trajectory& trajectory, double gamma, double lambda) {
    // Output buffers must be sized to match the trajectory.
    trajectory.advantages.assign(trajectory.size(), 0.0);
    trajectory.returns.assign(trajectory.size(), 0.0);

    // GAE accumulator (gae_{t+1} when we're computing step t). Iterating
    // backwards from the end lets us roll the recurrence with one variable.
    double gae = 0.0;

    // V(s_{t+1}) used in delta_t. For the very last step, "the state after the
    // last action" is the bootstrap_value the caller stored (V at the
    // post-rollout observation, or 0 if that step ended an episode).
    double next_value = trajectory.bootstrap_value;

    for (int t = static_cast<int>(trajectory.size()) - 1; t >= 0; --t) {
        const std::size_t index = static_cast<std::size_t>(t);

        // mask = 0 at an episode boundary kills the bootstrap and resets the
        // GAE accumulator — we must not credit reward forward across a reset.
        const double mask = trajectory.dones[index] ? 0.0 : 1.0;

        // TD residual delta_t = r_t + gamma * V(s_{t+1}) * mask - V(s_t).
        const double delta = trajectory.rewards[index] + (gamma * next_value * mask) - trajectory.values[index];

        // GAE recurrence (backwards): gae_t = delta_t + gamma * lambda * mask * gae_{t+1}.
        gae = delta + gamma * lambda * mask * gae;

        trajectory.advantages[index] = gae;

        // Returns target for the critic: A + V(s) = lambda-return target.
        // (When lambda=1 this collapses to the empirical sum of rewards.)
        trajectory.returns[index] = gae + trajectory.values[index];

        // For step t-1, "V at next state" is V(s_t).
        next_value = trajectory.values[index];
    }
}

// Standardize advantages to zero mean / unit std across the whole batch.
//
// Why: PPO is sensitive to advantage scale. With raw advantages, the gradient
// magnitude depends on the reward scale, which makes hyperparameters (esp.
// learning rate and clip range) brittle. Standardizing per-batch removes that
// coupling. We do this AFTER computing GAE — critic targets still use raw
// returns (since we want the value function to predict actual scale), but the
// policy gradient uses standardized advantages.
//
// The 1e-8 floor on variance prevents division-by-zero for degenerate batches.
template <typename Trajectory>
inline void normalize_advantages(Trajectory& trajectory) {
    if (trajectory.advantages.empty()) {
        return;
    }

    // One-pass mean/var (sum and sum-of-squares).
    double sum = 0.0;
    double sq_sum = 0.0;
    for (double advantage : trajectory.advantages) {
        sum += advantage;
        sq_sum += advantage * advantage;
    }

    const double count = static_cast<double>(trajectory.advantages.size());
    const double mean = sum / count;
    const double variance = std::max((sq_sum / count) - (mean * mean), 1e-8);
    const double stddev = std::sqrt(variance);

    for (double& advantage : trajectory.advantages) {
        advantage = (advantage - mean) / stddev;
    }
}

}  // namespace mujoco_rl_training
