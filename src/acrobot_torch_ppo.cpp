#include <envs/AcrobotEnv.hpp>
#include <envs/EnvTypes.hpp>
#include <mujoco_rl_training/rl/RunningMeanStd.h>
#include <mujoco_rl_training/rl/TrajectoryUtils.h>
#include <mujoco_rl_training/torch/Actor.h>
#include <mujoco_rl_training/torch/Critic.h>
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
    "artifacts/acrobot/torch_ppo/actor.pt",           "artifacts/acrobot/torch_ppo/critic.pt",
    "artifacts/acrobot/torch_ppo/log_std.pt",         "artifacts/acrobot/torch_ppo/metadata.txt",
    "artifacts/acrobot/torch_ppo/obs_normalizer.txt",
};

mujoco_rl_training::AcrobotEnvConfig make_env_config() {
    mujoco_rl_training::AcrobotEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/acrobot.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.tracked_frame_names = {"tcp_site"};
    // target_angles is still required by the constructor's validation, but unused in the phase-blended reward.
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {0.0, 0.0};   // unused; phase-blended reward uses link angles only
    config.target_link_angles = {kPi, kPi};
    // Balance-phase weights — get scaled by balance_weight (≈1 at upright, ≈0 far away) inside compute_reward.
    config.link_angle_cost_weights = {5.0, 5.0};
    config.velocity_cost_weights = {0.5, 0.5};
    config.control_cost_weights = {0.001};
    // Swing-up-phase weight — scaled by (1 - balance_weight) inside compute_reward.
    config.energy_weight = 0.05;
    config.phase_blend_sigma = 0.4;     // wider sigma = bigger "balance region"
    config.balance_bonus = 5.0;         // sparse alive reward — creates a discrete cliff at the sweet spot
    config.balance_radius = 0.15;       // link-error-norm threshold for the sweet spot
    config.balance_velocity_max = 1.5;  // per-joint velocity threshold (rad/s) for the sweet spot
    config.max_torques = {20.0};        // back to full authority for swing-up
    config.episode_horizon = 1000;
    config.repeat_action = 10;
    config.simulation_frequency = 1000;
    config.reset_angle_range = 0.35;
    config.reset_velocity_range = 0.5;
    return config;
}

constexpr double kInitialActionStd = 0.5;

std::vector<mujoco_rl_training::ResetSpace> make_training_scenarios() {
    return {
        // Near-upright with small velocities — pure stabilization practice
        {{kPi, 0.0}, {0.1, 0.1}, {0.0, 0.0}, {0.5, 0.5}},
        {{kPi, 0.0}, {0.2, 0.2}, {0.0, 0.0}, {1.0, 1.0}},
        {{kPi, 0.0}, {0.3, 0.3}, {0.0, 0.0}, {2.0, 2.0}},
        {{kPi, 0.0}, {0.5, 0.5}, {0.0, 0.0}, {3.0, 3.0}},  // catching from a fall
        {{kPi, 0.0}, {0.25, 0.25}, {0.0, 0.0}, {0.5, 0.5}},
        {{kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {1.5, 1.5}},
        {{0.5 * kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {4.0, 4.0}},
        {{-0.5 * kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {4.0, 4.0}},
        {{0.0, 0.0}, {0.35, 0.35}, {0.0, 0.0}, {2.0, 2.0}},
        {{0.0, 0.0}, {kPi, kPi}, {0.0, 0.0}, {4.0, 4.0}},
        // Eval start: hanging-down from rest, small jitter. Keeps train/eval distributions aligned.
        {{0.0, 0.0}, {0.05, 0.05}, {0.0, 0.0}, {0.0, 0.0}},
    };
}
}  // namespace

int main() {
    constexpr int kObsDim = 6;
    constexpr int kActionDim = 1;
    constexpr int kEpochs = 1000;
    constexpr std::size_t kStepsPerEpoch = 4000;
    constexpr int kEpisodesPerEvaluation = 10;
    constexpr double kGamma = 0.99;
    constexpr double kLambda = 0.95;
    constexpr double kRewardScale = 0.01;
    constexpr int kPpoTrainIters = 10;
    constexpr int kMiniBatchSize = 512;
    constexpr double kClipEpsilon = 0.2;
    constexpr double kTargetKl = 0.02;
    constexpr double kEntropyCoef = 0.001;
    constexpr int kLogEvery = 10;
    constexpr double kCheckpointImprovementThreshold = 0.0;
    constexpr int kEarlyStopPatience = 200;

    torch::manual_seed(0);

    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    auto config = make_env_config();
    const auto training_scenarios = make_training_scenarios();
    const mujoco_rl_training::ResetSpace evaluation_scenario{{0.0, 0.0}, {0.05, 0.05}, {0.0, 0.0}, {0.0, 0.0}};
    std::mt19937 reset_rng(0);

    mujoco_rl_training::AcrobotEnv env(config);
    mujoco_rl_training::TorchActor actor(kObsDim, kActionDim, kInitialActionStd);
    mujoco_rl_training::TorchCritic critic(kObsDim);
    mujoco_rl_training::RunningMeanStd obs_normalizer(kObsDim);

    actor->to(device);
    critic->to(device);

    torch::optim::Adam actor_optimizer(actor->parameters(), torch::optim::AdamOptions(0.0003));
    torch::optim::Adam critic_optimizer(critic->parameters(), torch::optim::AdamOptions(0.001));

    auto train_reset = [&](auto& env) {
        return mujoco_rl_training::reset_from_random_scenario(env, training_scenarios, reset_rng);
    };

    auto eval_reset = [&](auto& env) { return env.reset(evaluation_scenario); };

    auto eval_stats = mujoco_rl_training::evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, eval_reset,
                                                               &obs_normalizer);
    double best_mean_return = eval_stats.mean_return;
    int epochs_without_improvement = 0;
    mujoco_rl_training::save_acrobot_torch_actor_critic_policy(
        actor, critic, actor->log_std, kArtifactPaths, env.config(), best_mean_return, -1, kEpisodesPerEvaluation,
        "acrobot_torch_ppo", &obs_normalizer);

    std::cout << "Initial mean-policy return: " << best_mean_return << '\n';

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        auto batch = mujoco_rl_training::collect_ppo_batch(env, actor, critic, device, kStepsPerEpoch, kRewardScale,
                                                           train_reset, &obs_normalizer);

        mujoco_rl_training::compute_gae(batch, kGamma, kLambda);
        mujoco_rl_training::normalize_advantages(batch);

        const auto stats =
            mujoco_rl_training::update_ppo(actor, critic, actor_optimizer, critic_optimizer, batch, device,
                                           kPpoTrainIters, kMiniBatchSize, kClipEpsilon, kTargetKl, kEntropyCoef);

        eval_stats = mujoco_rl_training::evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, eval_reset,
                                                              &obs_normalizer);

        if (eval_stats.mean_return > best_mean_return + kCheckpointImprovementThreshold) {
            best_mean_return = eval_stats.mean_return;
            epochs_without_improvement = 0;
            mujoco_rl_training::save_acrobot_torch_actor_critic_policy(
                actor, critic, actor->log_std, kArtifactPaths, env.config(), best_mean_return, epoch,
                kEpisodesPerEvaluation, "acrobot_torch_ppo", &obs_normalizer);
        } else {
            ++epochs_without_improvement;
        }

        if (epoch % kLogEvery == 0) {
            std::cout << "epoch=" << epoch << " mean_policy_return=" << eval_stats.mean_return
                      << " best_mean_return=" << best_mean_return << " no_improve=" << epochs_without_improvement
                      << " actor_loss=" << stats.actor_loss << " critic_loss=" << stats.critic_loss
                      << " approx_kl=" << stats.approx_kl << " clip_fraction=" << stats.clip_fraction
                      << " action_std=" << stats.mean_std << " entropy=" << stats.entropy << '\n';
        }

        if (epochs_without_improvement >= kEarlyStopPatience) {
            std::cout << "Early stopping after " << epochs_without_improvement << " epochs without improvement.\n";
            break;
        }
    }

    std::cout << "Final best mean-policy return: " << best_mean_return << '\n';
    std::cout << "Saved best actor to: " << kArtifactPaths.actor_path << '\n';
    std::cout << "Saved best critic to: " << kArtifactPaths.critic_path << '\n';
    std::cout << "Saved log_std to: " << kArtifactPaths.log_std_path << '\n';
    std::cout << "Saved metadata to: " << kArtifactPaths.metadata_path << '\n';
    std::cout << "Saved obs normalizer to: " << kArtifactPaths.obs_normalizer_path << '\n';
    return 0;
}
