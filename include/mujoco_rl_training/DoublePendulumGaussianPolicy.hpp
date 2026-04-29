#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

namespace mujoco_rl_training {

struct DoublePendulumGaussianPolicy {
    static constexpr std::size_t kObservationDim = 6;
    static constexpr std::size_t kActionDim = 2;

    std::vector<double> mean_action(const std::vector<double>& observations) const {
        validate_dimensions(observations);
        std::vector<double> action = bias;
        for (std::size_t i = 0; i < observations.size(); ++i) {
            action[0] += observations[i] * weights[0][i];
            action[1] += observations[i] * weights[1][i];
        }
        return action;
    }

    std::vector<double> sample_action(const std::vector<double>& observations, std::mt19937& rng) const {
        const auto mu = mean_action(observations);
        std::normal_distribution<double> dist_0(mu[0], sigma);
        std::normal_distribution<double> dist_1(mu[1], sigma);
        std::vector<double> dist{dist_0(rng), dist_1(rng)};
        return dist;
    }
    double log_probability(const std::vector<double>& observations, const std::vector<double>& action) const {
        validate_dimensions(observations);
        if (action.size() != kActionDim) {
            throw std::runtime_error("DoublePendulumGaussianPolicy expected action size 2");
        }
        const auto mu = mean_action(observations);
        const double variance = sigma * sigma;
        constexpr double kPi = 3.14159265358979323846;
        double log_prob = 0.0;
        for (std::size_t i = 0; i < kActionDim; ++i) {
            const double diff = mu[i] - action[i];
            log_prob += -0.5 * std::log(2.0 * kPi * variance) - (diff * diff) / (2.0 * variance);
        }
        return log_prob;
    }

    std::vector<std::vector<double>> weights =
        std::vector<std::vector<double>>(kActionDim, std::vector<double>(kObservationDim, 0.0));
    std::vector<double> bias = std::vector<double>(kActionDim, 0.0);
    double sigma = 0.5;

   private:
    void validate_dimensions(const std::vector<double>& observations) const {
        if (observations.size() != kObservationDim) {
            throw std::runtime_error("DoublePendulumGaussianPolicy expected observation size 6");
        }
        if (weights.size() != kActionDim || bias.size() != kActionDim) {
            throw std::runtime_error("DoublePendulumGaussianPolicy expected 2 action outputs");
        }
        for (const auto& action_weights : weights) {
            if (action_weights.size() != kObservationDim) {
                throw std::runtime_error("DoublePendulumGaussianPolicy expected 6 weights per action");
            }
        }
        if (sigma <= 0.0) {
            throw std::runtime_error("DoublePendulumGaussianPolicy sigma must be positive");
        }
    }
};

struct DoublePendulumValueFunction {
    static constexpr std::size_t kObservationDim = DoublePendulumGaussianPolicy::kObservationDim;

    double predict(const std::vector<double>& observations) const {
        validate_dimensions(observations);
        double value = bias;
        for (std::size_t i = 0; i < observations.size(); ++i) {
            value += observations[i] * weights[i];
        }
        return value;
    }

    std::vector<double> weights = std::vector<double>(kObservationDim, 0.0);
    double bias = 0.0;

   private:
    void validate_dimensions(const std::vector<double>& observations) const {
        if (observations.size() != kObservationDim) {
            throw std::runtime_error("DoublePendulumValueFunction expected observation size 6");
        }
        if (weights.size() != kObservationDim) {
            throw std::runtime_error("DoublePendulumValueFunction expected 6 weights");
        }
    }
};
}  // namespace mujoco_rl_training
