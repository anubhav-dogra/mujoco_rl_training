#include <envs/DoublePendulumEnv.h>
#include <mujoco_rl_training/rl/RunningMeanStd.h>
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
#include <envs/EnvTypes.hpp>
namespace {

constexpr double kPi = 3.14159265358979323846;
const mujoco_rl_training::TorchActorCriticArtifactPaths kArtifactPaths{
    "artifacts/double_pendulum_torch_ppo_actor.pt",           "artifacts/double_pendulum_torch_ppo_critic.pt",
    "artifacts/double_pendulum_torch_ppo_log_std.pt",         "artifacts/double_pendulum_torch_ppo.meta.txt",
    "artifacts/double_pendulum_torch_ppo_obs_normalizer.txt",
};

constexpr double kInitialActionStd = 0.7;

mujoco_rl_training::DoublePendulumEnvConfig make_env_config() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.target_angles = {kPi, 0.0};
    // Halved relative to the prior {1.0, 1.0} since target_link_angles enforces the same upright constraint.
    config.angle_cost_weights = {0.5, 0.5};
    config.target_link_angles = {kPi, kPi};
    config.link_angle_cost_weights = {0.5, 0.5};
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

std::vector<mujoco_rl_training::ResetSpace> make_training_scenarios() {
    return {{{kPi, 0.0}, {0.25, 0.25}, {0.0, 0.0}, {0.5, 0.5}},
            {{kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {1.5, 1.5}},
            {{0.5 * kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {4.0, 4.0}},
            {{-0.5 * kPi, 0.0}, {0.6, 0.6}, {0.0, 0.0}, {4.0, 4.0}},
            {{0.0, 0.0}, {0.35, 0.35}, {0.0, 0.0}, {2.0, 2.0}},
            {{0.0, 0.0}, {kPi, kPi}, {0.0, 0.0}, {4.0, 4.0}},
            // Eval start: hanging-down from rest, small jitter. Keeps train/eval distributions aligned.
            {{0.0, 0.0}, {0.05, 0.05}, {0.0, 0.0}, {0.0, 0.0}}};
};

void apply_reward_schedule(mujoco_rl_training::DoublePendulumEnv& env, int epoch) {
    // Angle and link-angle weights are halved relative to the original schedule to avoid double-dipping
    // (both target_angles and target_link_angles encode the same upright constraint).
    if (epoch < 25) {
        env.set_training_weights({2.5, 1.5}, {0.08, 0.1}, {0.001, 0.0015}, {20.0, 12.0}, {0.5, 1.25});
        return;
    }

    if (epoch < 80) {
        env.set_training_weights({2.0, 2.0}, {0.1, 0.12}, {0.0012, 0.0018}, {18.0, 12.0}, {0.5, 1.0});
        return;
    }

    env.set_training_weights({1.5, 2.0}, {0.12, 0.12}, {0.0015, 0.0018}, {16.0, 12.0}, {0.5, 1.0});
}

void apply_final_evaluation_weights(mujoco_rl_training::DoublePendulumEnv& env) {
    env.set_training_weights({1.5, 2.0}, {0.12, 0.12}, {0.0015, 0.0018}, {16.0, 12.0}, {0.5, 1.0});
}

}  // namespace

int main() {
    constexpr int kObsDim = 6;
    constexpr int kActionDim = 2;
    constexpr int kEpochs = 500;
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
    constexpr int kLogEvery = 5;
    constexpr double kCheckpointImprovementThreshold = 0.0;
    constexpr int kEarlyStopPatience = 200;

    torch::manual_seed(0);
    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    auto config = make_env_config();
    const auto training_scenarios = make_training_scenarios();
    const mujoco_rl_training::ResetSpace evaluation_scenario{{0.0, 0.0}, {0.05, 0.05}, {0.0, 0.0}, {0.0, 0.0}};
    std::mt19937 reset_rng(0);
    mujoco_rl_training::DoublePendulumEnv env(config);

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

    apply_final_evaluation_weights(env);
    auto eval_stats = mujoco_rl_training::evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, eval_reset,
                                                               &obs_normalizer);
    double best_mean_return = eval_stats.mean_return;
    int epochs_without_improvement = 0;
    mujoco_rl_training::save_double_pendulum_torch_actor_critic_policy(
        actor, critic, actor->log_std, kArtifactPaths, env.config(), best_mean_return, -1, kEpisodesPerEvaluation,
        "torch_ppo_actor_critic", &obs_normalizer);
    std::cout << "Initial mean-policy return: " << best_mean_return << '\n';

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        apply_reward_schedule(env, epoch);

        auto batch = mujoco_rl_training::collect_ppo_batch(env, actor, critic, device, kStepsPerEpoch, kRewardScale,
                                                           train_reset, &obs_normalizer);
        mujoco_rl_training::compute_gae(batch, kGamma, kLambda);
        mujoco_rl_training::normalize_advantages(batch);
        const auto stats =
            mujoco_rl_training::update_ppo(actor, critic, actor_optimizer, critic_optimizer, batch, device,
                                           kPpoTrainIters, kMiniBatchSize, kClipEpsilon, kTargetKl, kEntropyCoef);

        apply_final_evaluation_weights(env);
        eval_stats = mujoco_rl_training::evaluate_mean_policy(env, actor, device, kEpisodesPerEvaluation, eval_reset,
                                                              &obs_normalizer);
        if (eval_stats.mean_return > best_mean_return + kCheckpointImprovementThreshold) {
            best_mean_return = eval_stats.mean_return;
            epochs_without_improvement = 0;
            mujoco_rl_training::save_double_pendulum_torch_actor_critic_policy(
                actor, critic, actor->log_std, kArtifactPaths, env.config(), best_mean_return, epoch,
                kEpisodesPerEvaluation, "torch_ppo_actor_critic", &obs_normalizer);
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
