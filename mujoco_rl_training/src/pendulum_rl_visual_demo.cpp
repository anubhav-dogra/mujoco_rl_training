#include <mujoco_rl_training/PendulumEnv.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <mujoco/mujoco.h>
#include <mujoco_viewer/MujocoViewer.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

int main() {
    mujoco_rl_training::PendulumEnvConfig config;
    config.xml_path = ament_index_cpp::get_package_share_directory("mujoco_models") + "/models/pendulum/pendulum.xml";
    config.simulation_frequency = 1000;
    config.max_torque = 20.0;
    config.episode_horizon = 2000;
    config.seed = 0;

    mujoco_rl_training::PendulumEnv env(config);
    auto observation = env.reset();

    std::cout << "Initial observation: ["
              << observation[0] << ", "
              << observation[1] << ", "
              << observation[2] << "]\n";

    auto& sim_core = env.sim_core();
    std::atomic<bool> paused{false};
    std::atomic<bool> reset_requested{false};

    MujocoViewer viewer;
    viewer.initialize(*sim_core.model(), false, &paused, &reset_requested);
    viewer.set_camera_properties({0.0, 0.0, 0.5}, 2.5, 135.0, -30.0, false);

    std::unique_ptr<mjData, void (*)(mjData*)> render_data(mj_makeData(sim_core.model()), mj_deleteData);
    if (!render_data) {
        std::cerr << "Failed to allocate MuJoCo render snapshot.\n";
        return EXIT_FAILURE;
    }

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> action_dist(-config.max_torque, config.max_torque);

    const auto frame_period =
        std::chrono::duration<double>(1.0 / static_cast<double>(config.simulation_frequency));

    for (int step = 0; !viewer.should_close(); ++step) {
        const auto frame_start = std::chrono::steady_clock::now();

        if (reset_requested.exchange(false, std::memory_order_acq_rel)) {
            observation = env.reset();
            std::cout << "Reset observation: ["
                      << observation[0] << ", "
                      << observation[1] << ", "
                      << observation[2] << "]\n";
        }

        if (!paused.load(std::memory_order_acquire)) {
            const double action = action_dist(rng);
            const auto result = env.step(action);
            observation = result.observation;

            if (result.terminated || result.truncated) {
                observation = env.reset();
                std::cout << "Episode reset observation: ["
                          << observation[0] << ", "
                          << observation[1] << ", "
                          << observation[2] << "]\n";
            }
        }

        {
            std::lock_guard<std::recursive_mutex> lock(sim_core.state_mutex());
            mj_copyData(render_data.get(), sim_core.model(), sim_core.data());
        }

        viewer.update_scene(*render_data);
        viewer.present();

        std::this_thread::sleep_until(frame_start + frame_period);
    }

    return EXIT_SUCCESS;
}
