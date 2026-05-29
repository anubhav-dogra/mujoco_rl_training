#include <envs/AcrobotEnv.hpp>
#include <mujoco_rl_training/AcrobotPolicyMetadata.h>
#include <mujoco_rl_training/VisualDemoUtils.h>
#include <mujoco_rl_training/rl/RunningMeanStd.h>
#include <mujoco_rl_training/torch/Actor.h>
#include <mujoco_rl_training/torch/GaussianPolicy.h>
#include <mujoco_rl_training/torch/TensorUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <torch/torch.h>

#include <iostream>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;
const char* kDefaultActorArtifactPath = "artifacts/acrobot/torch_ppo/actor.pt";
const char* kDefaultMetadataArtifactPath = "artifacts/acrobot/torch_ppo/metadata.txt";
const char* kDefaultObsNormalizerArtifactPath = "artifacts/acrobot/torch_ppo/obs_normalizer.txt";

mujoco_rl_training::AcrobotEnvConfig make_env_config() {
    mujoco_rl_training::AcrobotEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/acrobot.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.tracked_frame_names = {"tcp_site"};
    config.target_angles = {kPi, 0.0};
    config.angle_cost_weights = {1.0, 1.0};
    config.target_link_angles = {kPi, kPi};
    config.link_angle_cost_weights = {1.0, 1.0};
    config.velocity_cost_weights = {1.0, 1.0};
    config.control_cost_weights = {0.01};
    config.max_torques = {12.0};
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
    constexpr int kActionDim = 1;

    const std::string actor_path = (argc > 1) ? argv[1] : kDefaultActorArtifactPath;
    const std::string metadata_path = (argc > 2) ? argv[2] : kDefaultMetadataArtifactPath;
    const std::string obs_normalizer_path = (argc > 3) ? argv[3] : kDefaultObsNormalizerArtifactPath;
    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    auto config = make_env_config();
    const auto metadata = mujoco_rl_training::load_acrobot_policy_metadata(metadata_path);
    if (metadata.has_value()) {
        config = metadata->config;
        config.tracked_frame_names = {"tcp_site"};
        std::cout << "Loaded policy metadata from: " << metadata_path << '\n'
                  << "Saved policy best return: " << metadata->best_return << '\n'
                  << "Best epoch: " << metadata->num_iterations << '\n'
                  << "Episodes per evaluation: " << metadata->episodes_per_evaluation << '\n'
                  << "Saved action std dev: " << metadata->noise_std_dev << '\n';
    } else {
        std::cout << "Policy metadata not found, using rollout defaults: " << metadata_path << '\n';
    }

    mujoco_rl_training::AcrobotEnv env(config);

    mujoco_rl_training::TorchActor actor(kObsDim, kActionDim);
    torch::load(actor, actor_path);
    actor->to(device);
    actor->eval();

    auto obs_normalizer = mujoco_rl_training::RunningMeanStd::load(obs_normalizer_path);
    if (obs_normalizer.has_value()) {
        std::cout << "Loaded obs normalizer from: " << obs_normalizer_path << " (count=" << obs_normalizer->count()
                  << ")\n";
    } else {
        std::cout << "Obs normalizer not found at: " << obs_normalizer_path << " (will pass observations through)\n";
    }

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
            const auto normalized_obs =
                obs_normalizer.has_value() ? obs_normalizer->normalize(observation) : observation;
            const auto obs_tensor = mujoco_rl_training::vector_to_tensor(normalized_obs, device, true);
            const auto mean = actor->forward(obs_tensor);
            const auto normalized_action = mujoco_rl_training::squash_action(mean);
            return mujoco_rl_training::scale_action_from_action_space(
                mujoco_rl_training::tensor_to_vector(normalized_action.squeeze(0)), env.env_spec().action_limit_space);
        },
        {{0.0, 0.0, -0.7}, 3.0, 90.0, -10.0});
}
