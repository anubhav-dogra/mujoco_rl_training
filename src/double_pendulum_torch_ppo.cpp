#include <envs/DoublePendulumEnv.h>
#include <mujoco_rl_training/torch/GaussianPolicy.h>
#include <mujoco_rl_training/torch/Ppo.h>
#include <mujoco_rl_training/torch/TensorUtils.h>
#include <mujoco_rl_training/torch/TorchArtifacts.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const mujoco_rl_training::TorchActorCriticArtifactPaths kArtifactPaths{
    "artifacts/double_pendulum_torch_ppo_actor.pt",
    "artifacts/double_pendulum_torch_ppo_critic.pt",
    "artifacts/double_pendulum_torch_ppo_log_std.pt",
    "artifacts/double_pendulum_torch_ppo.meta.txt",
};

mujoco_rl_training::DoublePendulumEnvConfig make_env_config() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {1.0, 1.0};
    config.target_link_angles = {kPi, kPi};
    config.link_angle_cost_weights = {1.0, 1.0};
    config.velocity_cost_weights = {0.001, 0.01};
    config.control_cost_weights = {0.001, 0.0015};
    config.max_torques = {20.0, 12.0};
    config.episode_horizon = 1000;
    config.repeat_action = 10;
    config.simulation_frequency = 1000;
    config.reset_angle_range = 0.35;
    config.reset_velocity_range = 0.5;
    return config;
}

struct ResetScenario {
    std::vector<double> angle_centers;
    double angle_range = 0.0;
    double velocity_range = 0.0;
};

std::vector<ResetScenario> make_training_scenarios() {
    return {
        {{kPi, 0.0}, 0.25, 0.5},       {{kPi, 0.0}, 0.6, 1.5},  {{0.5 * kPi, 0.0}, 0.6, 4.0},
        {{-0.5 * kPi, 0.0}, 0.6, 4.0}, {{0.0, 0.0}, 0.35, 2.0}, {{0.0, 0.0}, kPi, 4.0},
    };
}

std::vector<double> reset_from_scenario(mujoco_rl_training::DoublePendulumEnv& env, const ResetScenario& scenario) {
    return env.reset_around(scenario.angle_centers, scenario.angle_range, scenario.velocity_range);
}

std::vector<double> reset_from_random_scenario(mujoco_rl_training::DoublePendulumEnv& env,
                                               const std::vector<ResetScenario>& scenarios, std::mt19937& rng) {
    if (scenarios.empty()) {
        return env.reset();
    }

    std::uniform_int_distribution<std::size_t> scenario_dist(0, scenarios.size() - 1);
    return reset_from_scenario(env, scenarios[scenario_dist(rng)]);
}

void apply_reward_schedule(mujoco_rl_training::DoublePendulumEnv& env, int epoch) {
    if (epoch < 25) {
        env.set_training_weights({5.0, 3.0}, {0.08, 0.1}, {0.001, 0.0015}, {20.0, 12.0}, {1.0, 2.5});
        return;
    }

    if (epoch < 80) {
        env.set_training_weights({4.0, 4.0}, {0.1, 0.12}, {0.0012, 0.0018}, {18.0, 12.0}, {1.0, 2.0});
        return;
    }

    env.set_training_weights({3.0, 4.0}, {0.12, 0.12}, {0.0015, 0.0018}, {16.0, 12.0}, {1.0, 2.0});
}

void apply_final_evaluation_weights(mujoco_rl_training::DoublePendulumEnv& env) {
    env.set_training_weights({3.0, 4.0}, {0.12, 0.12}, {0.0015, 0.0018}, {16.0, 12.0}, {1.0, 2.0});
}

double scheduled_std(int epoch, int total_epochs) {
    constexpr double kStartStd = 0.7;
    constexpr double kEndStd = 0.25;
    const double progress =
        std::clamp(static_cast<double>(epoch) / static_cast<double>(std::max(total_epochs - 1, 1)), 0.0, 1.0);
    return kStartStd + (kEndStd - kStartStd) * progress;
}

double critic_value(mujoco_rl_training::TorchCritic& critic, const std::vector<double>& observation,
                    const torch::Device& device) {
    torch::NoGradGuard no_grad;
    const auto obs = mujoco_rl_training::vector_to_tensor(observation, device, true);
    return critic->forward(obs).item<double>();
}

mujoco_rl_training::PpoBatch collect_batch(mujoco_rl_training::DoublePendulumEnv& env,
                                           mujoco_rl_training::TorchActor& actor,
                                           mujoco_rl_training::TorchCritic& critic, const torch::Tensor& log_std,
                                           const torch::Device& device, std::size_t steps_per_epoch,
                                           double reward_scale, const std::vector<ResetScenario>& reset_scenarios,
                                           std::mt19937& reset_rng) {
    mujoco_rl_training::PpoBatch batch;
    batch.reserve(steps_per_epoch);

    auto observation = reset_from_random_scenario(env, reset_scenarios, reset_rng);
    std::vector<double> last_next_observation = observation;
    bool last_done = false;

    actor->eval();
    critic->eval();

    while (batch.size() < steps_per_epoch) {
        torch::NoGradGuard no_grad;
        const auto obs_tensor = mujoco_rl_training::vector_to_tensor(observation, device, true);
        const auto mean = actor->forward(obs_tensor);
        const auto raw_action = mujoco_rl_training::sample_gaussian_action(mean, log_std).detach();
        const auto normalized_action = mujoco_rl_training::squash_action(raw_action);
        const auto old_log_prob = mujoco_rl_training::squashed_gaussian_log_prob(raw_action, mean, log_std);
        const auto value = critic->forward(obs_tensor).item<double>();

        const auto action_command =
            mujoco_rl_training::scale_action(normalized_action.squeeze(0), env.config().max_torques);
        const auto result = env.step(action_command);
        const bool done = result.terminated || result.truncated;

        batch.store(observation, mujoco_rl_training::tensor_to_vector(raw_action.squeeze(0)),
                    old_log_prob.item<double>(), reward_scale * result.reward, value, done);
        last_next_observation = result.observation;
        last_done = done;
        observation = result.observation;

        if (done) {
            observation = reset_from_random_scenario(env, reset_scenarios, reset_rng);
        }
    }

    batch.bootstrap_value = last_done ? 0.0 : critic_value(critic, last_next_observation, device);
    actor->train();
    critic->train();

    return batch;
}

struct EvaluationStats {
    double mean_return = 0.0;
};

EvaluationStats evaluate_mean_policy(mujoco_rl_training::DoublePendulumEnv& env, mujoco_rl_training::TorchActor& actor,
                                     const torch::Device& device, int episodes, const ResetScenario& reset_scenario) {
    torch::NoGradGuard no_grad;
    actor->eval();

    double total_return = 0.0;

    for (int episode = 0; episode < episodes; ++episode) {
        auto observation = reset_from_scenario(env, reset_scenario);
        double episode_return = 0.0;

        while (true) {
            const auto obs_tensor = mujoco_rl_training::vector_to_tensor(observation, device, true);
            const auto mean = actor->forward(obs_tensor);
            const auto normalized_action = mujoco_rl_training::squash_action(mean);
            const auto action_command =
                mujoco_rl_training::scale_action(normalized_action.squeeze(0), env.config().max_torques);

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

}  // namespace

int main() {
    constexpr int kObsDim = 6;
    constexpr int kActionDim = 2;
    constexpr int kEpochs = 300;
    constexpr std::size_t kStepsPerEpoch = 4000;
    constexpr int kEpisodesPerEvaluation = 10;
    constexpr double kGamma = 0.99;
    constexpr double kLambda = 0.95;
    constexpr double kRewardScale = 0.01;
    constexpr int kPpoTrainIters = 10;
    constexpr int kMiniBatchSize = 512;
    constexpr double kClipEpsilon = 0.15;
    constexpr double kTargetKl = 0.015;
    constexpr int kLogEvery = 5;
    constexpr double kCheckpointImprovementThreshold = 0.0;
    constexpr int kEarlyStopPatience = 50;

    torch::manual_seed(0);
    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    auto config = make_env_config();
    const auto training_scenarios = make_training_scenarios();
    const ResetScenario evaluation_scenario{{0.0, 0.0}, 0.05, 0.0};
    std::mt19937 reset_rng(0);
    mujoco_rl_training::DoublePendulumEnv env(config);

    mujoco_rl_training::TorchActor actor(kObsDim, kActionDim);
    mujoco_rl_training::TorchCritic critic(kObsDim);
    actor->to(device);
    critic->to(device);

    auto log_std =
        torch::full({kActionDim}, std::log(scheduled_std(0, kEpochs)), torch::TensorOptions().dtype(torch::kFloat32))
            .to(device);

    torch::optim::Adam actor_optimizer(actor->parameters(), torch::optim::AdamOptions(0.0003));
    torch::optim::Adam critic_optimizer(critic->parameters(), torch::optim::AdamOptions(0.001));

    apply_final_evaluation_weights(env);
    auto eval_stats = evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, evaluation_scenario);
    double best_mean_return = eval_stats.mean_return;
    int epochs_without_improvement = 0;
    mujoco_rl_training::save_double_pendulum_torch_actor_critic_policy(
        actor, critic, log_std, kArtifactPaths, env.config(), best_mean_return, -1, kEpisodesPerEvaluation,
        "torch_ppo_actor_critic");
    std::cout << "Initial mean-policy return: " << best_mean_return << '\n';

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        apply_reward_schedule(env, epoch);
        const double current_std = scheduled_std(epoch, kEpochs);
        log_std =
            torch::full({kActionDim}, std::log(current_std), torch::TensorOptions().dtype(torch::kFloat32)).to(device);

        auto batch = collect_batch(env, actor, critic, log_std, device, kStepsPerEpoch, kRewardScale,
                                   training_scenarios, reset_rng);
        mujoco_rl_training::compute_gae(batch, kGamma, kLambda);
        mujoco_rl_training::normalize_advantages(batch);
        const auto stats =
            mujoco_rl_training::update_ppo(actor, critic, actor_optimizer, critic_optimizer, log_std, batch, device,
                                           kPpoTrainIters, kMiniBatchSize, kClipEpsilon, kTargetKl);

        apply_final_evaluation_weights(env);
        eval_stats = evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, evaluation_scenario);
        if (eval_stats.mean_return > best_mean_return + kCheckpointImprovementThreshold) {
            best_mean_return = eval_stats.mean_return;
            epochs_without_improvement = 0;
            mujoco_rl_training::save_double_pendulum_torch_actor_critic_policy(
                actor, critic, log_std, kArtifactPaths, env.config(), best_mean_return, epoch, kEpisodesPerEvaluation,
                "torch_ppo_actor_critic");
        } else {
            ++epochs_without_improvement;
        }

        if (epoch % kLogEvery == 0) {
            std::cout << "epoch=" << epoch << " mean_policy_return=" << eval_stats.mean_return
                      << " best_mean_return=" << best_mean_return << " no_improve=" << epochs_without_improvement
                      << " actor_loss=" << stats.actor_loss << " critic_loss=" << stats.critic_loss
                      << " approx_kl=" << stats.approx_kl << " clip_fraction=" << stats.clip_fraction
                      << " action_std=" << current_std << '\n';
        }

        if (epochs_without_improvement >= kEarlyStopPatience) {
            std::cout << "Early stopping after " << epochs_without_improvement
                      << " epochs without final-objective improvement.\n";
            break;
        }
    }

    std::cout << "Final best mean-policy return: " << best_mean_return << '\n';
    std::cout << "Saved best actor to: " << kArtifactPaths.actor_path << '\n';
    std::cout << "Saved best critic to: " << kArtifactPaths.critic_path << '\n';
    std::cout << "Saved log_std to: " << kArtifactPaths.log_std_path << '\n';
    std::cout << "Saved metadata to: " << kArtifactPaths.metadata_path << '\n';
    return 0;
}
