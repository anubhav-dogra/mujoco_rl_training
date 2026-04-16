#pragma once
#include <array>
#include <cmath>
#include <random>
#include <vector>

namespace mujoco_rl_training {

struct TrajectoryStep {
    std::array<double, 3> observation{};
    double sampled_action = 0.0;
    double reward = 0.0;
    double log_probability = 0.0;
};
struct Trajectory {
    std::vector<TrajectoryStep> steps;
};

struct PendulumGaussianPolicy {
    double mean_action(const std::array<double, 3>& observations) const {
        return (weights[0] * observations[0]) + (weights[1] * observations[1]) + (weights[2] * observations[2]) + bias;
    }

    double sample_action(const std::array<double, 3>& observations, std::mt19937& rng) const {
        const double mu = mean_action(observations);
        std::normal_distribution<double> dist(mu, sigma);
        return dist(rng);
    }

    /* The Gaussian density is:

    pi(a|s) = 1 / sqrt(2*pi*sigma^2) * exp( -(a - mu)^2 / (2*sigma^2) )

    Take log:

    log pi(a|s) =
    -0.5 * log(2*pi*sigma^2)
    - (a - mu)^2 / (2*sigma^2)
  */

    double log_probability(const std::array<double, 3>& observations, double action) const {
        constexpr double kPi = 3.14159265358979323846;
        const double mu = mean_action(observations);
        const double variance = sigma * sigma;
        const double difference = action - mu;
        return -0.5 * std::log(2.0 * kPi * variance) - (difference * difference) / (2.0 * variance);
    }

    std::array<double, 3> weights{};
    double bias = 0.0;
    double sigma = 0.1;
};
}  // namespace mujoco_rl_training
