#include <envs/DoublePendulumEnv.h>
#include <mujoco_rl_training/DoublePendulumPolicyMetadata.h>
#include <mujoco_rl_training/VisualDemoUtils.h>
#include <mujoco_rl_training/torch/Actor.h>
#include <mujoco_rl_training/torch/GaussianPolicy.h>
#include <mujoco_rl_training/torch/TensorUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kDefaultActorArtifactPath = "artifacts/double_pendulum_torch_vpg_actor.pt";
const char* kDefaultMetadataArtifactPath = "artifacts/double_pendulum_torch_vpg.meta.txt";

mujoco_rl_training::DoublePendulumEnvConfig make_env_config() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {3.0, 2.0};
    config.velocity_cost_weights = {0.001, 0.001};
    config.control_cost_weights = {0.0001, 0.0001};
    config.max_torques = {20.0, 15.0};
    config.episode_horizon = 1000;
    config.repeat_action = 10;
    config.simulation_frequency = 1000;
    config.reset_angle_range = 0.35;
    config.reset_velocity_range = 0.5;
    return config;
}

}  // namespace

int main(int argc, char* argv[]) {
    constexpr int kObsDim = 6;
    constexpr int kActionDim = 2;

    const std::string actor_path = (argc > 1) ? argv[1] : kDefaultActorArtifactPath;
    const std::string metadata_path = (argc > 2) ? argv[2] : kDefaultMetadataArtifactPath;
    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    auto config = make_env_config();
    const auto metadata = mujoco_rl_training::load_double_pendulum_policy_metadata(metadata_path);
    if (metadata.has_value()) {
        config = metadata->config;
        std::cout << "Loaded policy metadata from: " << metadata_path << '\n'
                  << "Saved policy best return: " << metadata->best_return << '\n'
                  << "Best epoch: " << metadata->num_iterations << '\n'
                  << "Episodes per evaluation: " << metadata->episodes_per_evaluation << '\n'
                  << "Saved action std dev: " << metadata->noise_std_dev << '\n';
    } else {
        std::cout << "Policy metadata not found, using rollout defaults: " << metadata_path << '\n';
    }

    mujoco_rl_training::DoublePendulumEnv env(config);

    mujoco_rl_training::TorchActor actor(kObsDim, kActionDim);
    torch::load(actor, actor_path);
    actor->to(device);
    actor->eval();

    auto observation = env.reset();
    std::cout << "Loaded Torch actor from: " << actor_path << '\n';
    std::cout << "Initial observation:";
    for (double value : observation) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';

    return mujoco_rl_training::run_visual_demo(
        env, observation,
        [&]() {
            torch::NoGradGuard no_grad;
            const auto obs_tensor = mujoco_rl_training::vector_to_tensor(observation, device, true);
            const auto mean = actor->forward(obs_tensor);
            const auto normalized_action = mujoco_rl_training::squash_action(mean);
            return mujoco_rl_training::scale_action(normalized_action.squeeze(0), env.config().max_torques);
        },
        {{0.0, 0.0, 0.5}, 3.5, 90.0, 0.0});
}
