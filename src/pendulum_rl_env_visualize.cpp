#include <mujoco_rl_training/PendulumEnv.h>
#include <mujoco_rl_training/VisualDemoUtils.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <random>

int main() {
    mujoco_rl_training::PendulumEnvConfig config;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";
    config.simulation_frequency = 1000;
    config.max_torque = 40.0;
    config.episode_horizon = 200;
    config.seed = 0;
    config.repeat_action = 20;

    mujoco_rl_training::PendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation: [" << observation[0] << ", " << observation[1] << ", " << observation[2]
              << "]\n";

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> action_dist(-config.max_torque, config.max_torque);
    return mujoco_rl_training::run_visual_demo(
        env, observation, [&]() { return action_dist(rng); }, {{0.0, 0.0, 0.5}, 2.5, 135.0, -30.0});
}
