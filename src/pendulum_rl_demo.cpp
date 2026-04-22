#include <mujoco_rl_training/PendulumEnv.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

int main() {
    mujoco_rl_training::PendulumEnvConfig config;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";
    config.simulation_frequency = 1000;
    config.max_torque = 40.0;
    config.episode_horizon = 200;
    config.seed = 0;
    config.repeat_action = 20;

    mujoco_rl_training::PendulumEnv env(config);
    const auto initial_observation = env.reset();

    std::cout << "Initial observation: [" << initial_observation[0] << ", " << initial_observation[1] << ", "
              << initial_observation[2] << "]\n";

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> action_dist(-config.max_torque, config.max_torque);

    for (int step = 0; step < 20; ++step) {
        const double action = action_dist(rng);
        const auto result = env.step(action);

        std::cout << "step=" << step << " action=" << action << " reward=" << result.reward << " obs=["
                  << result.observation[0] << ", " << result.observation[1] << ", " << result.observation[2] << "]"
                  << " truncated=" << (result.truncated ? "true" : "false") << "\n";

        if (result.terminated || result.truncated) {
            break;
        }
    }

    return EXIT_SUCCESS;
}
