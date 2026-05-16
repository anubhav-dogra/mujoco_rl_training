#pragma once

#include <mujoco_rl_training/DoublePendulumPolicyMetadata.h>
#include <mujoco_rl_training/PolicyIO.h>
#include <mujoco_rl_training/torch/Actor.h>
#include <mujoco_rl_training/torch/Critic.h>

#include <string>

#include <torch/torch.h>

namespace mujoco_rl_training {

struct TorchActorCriticArtifactPaths {
    std::string actor_path;
    std::string critic_path;
    std::string log_std_path;
    std::string metadata_path;
};

inline void save_torch_actor_critic_artifacts(TorchActor& actor, TorchCritic& critic, const torch::Tensor& log_std,
                                              const TorchActorCriticArtifactPaths& paths) {
    ensure_artifact_parent_directory(paths.actor_path);
    ensure_artifact_parent_directory(paths.critic_path);
    ensure_artifact_parent_directory(paths.log_std_path);

    torch::save(actor, paths.actor_path);
    torch::save(critic, paths.critic_path);
    torch::save(log_std.to(torch::kCPU), paths.log_std_path);
}

inline void append_torch_actor_critic_metadata(const TorchActorCriticArtifactPaths& paths,
                                               const std::string& policy_type, int epoch) {
    auto metadata = open_artifact_append(paths.metadata_path);
    metadata << "policy_type=" << policy_type << '\n';
    metadata << "actor_path=" << paths.actor_path << '\n';
    metadata << "critic_path=" << paths.critic_path << '\n';
    metadata << "log_std_path=" << paths.log_std_path << '\n';
    metadata << "best_epoch=" << epoch << '\n';
}

inline void save_double_pendulum_torch_actor_critic_policy(
    TorchActor& actor, TorchCritic& critic, const torch::Tensor& log_std, const TorchActorCriticArtifactPaths& paths,
    const DoublePendulumEnvConfig& config, double best_mean_return, int epoch, int episodes_per_evaluation,
    const std::string& policy_type) {
    save_torch_actor_critic_artifacts(actor, critic, log_std, paths);

    const double mean_std = torch::exp(log_std).mean().item<double>();
    save_double_pendulum_policy_metadata(paths.metadata_path, paths.actor_path, config, best_mean_return, epoch,
                                         episodes_per_evaluation, mean_std);
    append_torch_actor_critic_metadata(paths, policy_type, epoch);
}

}  // namespace mujoco_rl_training
