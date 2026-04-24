#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_rl_training/VisualDemoUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <random>
#include <vector>

int main() {
    mujoco_rl_training::DoublePendulumEnvConfig config;
    config.xml_path =
        ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/double_pendulum/double_pendulum.xml";
    config.joint_names = {"joint_1", "joint_2"};
    config.max_torques = {30.0, 15.0};
    config.target_angles = {3.14159265358979323846, 3.14159265358979323846};
    config.angle_cost_weights = {1.0, 1.0};
    config.velocity_cost_weights = {0.1, 0.1};
    config.control_cost_weights = {0.001, 0.001};
    config.episode_horizon = 2000;
    config.repeat_action = 20;

    mujoco_rl_training::DoublePendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation:";
    for (double value : observation) {
        std::cout << ' ' << value;
    }
    std::cout << std::endl;

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> action_1_dist(-config.max_torques[0], config.max_torques[0]);
    std::uniform_real_distribution<double> action_2_dist(-config.max_torques[1], config.max_torques[1]);
    return mujoco_rl_training::run_visual_demo(
        env, observation, [&]() { return std::vector<double>{action_1_dist(rng), action_2_dist(rng)}; },
        {{0.0, 0.0, -0.6}, 3.0, 135.0, -20.0});
}
