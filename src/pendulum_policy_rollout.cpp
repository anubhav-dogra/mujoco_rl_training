#include <envs/PendulumEnv.h>
#include <mujoco_rl_training/PendulumPolicyMetadata.h>
#include <mujoco_rl_training/VisualDemoUtils.h>
#include <mujoco_rl_training/artifacts/PolicyLoaders.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <iostream>

namespace {

const char* kDefaultPolicyArtifactPath = "artifacts/pendulum_best_policy.txt";

}  // namespace

int main(int argc, char* argv[]) {
    mujoco_rl_training::PendulumEnvConfig config;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";
    config.simulation_frequency = 1000;
    config.max_torque = 20.0;
    config.episode_horizon = 400;
    config.seed = 0;
    config.repeat_action = 20;
    config.reset_angle_range = 2.0;
    config.reset_velocity_range = 0.5;

    const std::string policy_path = (argc > 1) ? argv[1] : kDefaultPolicyArtifactPath;
    const std::string metadata_path =
        (argc > 2) ? argv[2] : mujoco_rl_training::pendulum_metadata_path_for_policy(policy_path);
    auto action_scale = mujoco_rl_training::PendulumPolicyActionScale::PhysicalTorque;
    const auto metadata = mujoco_rl_training::load_pendulum_policy_metadata(metadata_path);
    if (metadata.has_value()) {
        config = metadata->config;
        action_scale = metadata->action_scale;
        std::cout << "Loaded policy metadata from: " << metadata_path << "\n";
        std::cout << "Policy algorithm: " << metadata->algorithm << "\n";
        std::cout << "Policy action scale: " << mujoco_rl_training::to_string(action_scale) << "\n";
        if (metadata->best_return.has_value()) {
            std::cout << "Saved policy best_return: " << metadata->best_return.value() << "\n";
        }
    } else {
        std::cout << "Policy metadata not found, assuming physical torque actions: " << metadata_path << "\n";
    }

    const auto loaded_policy = mujoco_rl_training::load_pendulum_policy(policy_path);
    const mujoco_rl_training::PendulumLinearPolicy& policy = loaded_policy.policy;
    mujoco_rl_training::PendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation: [" << observation[0] << ", " << observation[1] << ", " << observation[2]
              << "]\n";
    std::cout << "Loaded policy from: " << policy_path << "\n";
    if (loaded_policy.has_sigma) {
        std::cout << "Policy artifact type: Gaussian mean-policy replay (sigma=" << loaded_policy.sigma << ")\n";
    } else {
        std::cout << "Policy artifact type: deterministic linear policy\n";
    }
    std::cout << "Policy weights: [" << policy.weights[0] << ", " << policy.weights[1] << ", " << policy.weights[2]
              << "] bias=" << policy.bias << "\n";

    return mujoco_rl_training::run_visual_demo(
        env, observation,
        [&]() {
            const double policy_action = policy.action_from_obs(observation);
            return mujoco_rl_training::pendulum_physical_action(policy_action, action_scale, config.max_torque);
        },
        {{0.0, 0.0, 0.5}, 2.5, 135.0, -30.0});
}
