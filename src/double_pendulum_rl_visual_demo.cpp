#include <mujoco_rl_training/DoublePendulumEnv.h>

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

    auto& sim_core = env.sim_core();
    std::atomic<bool> paused{false};
    std::atomic<bool> reset_requested{false};

    MujocoViewer viewer;
    viewer.initialize(*sim_core.model(), false, &paused, &reset_requested);
    viewer.set_camera_properties({0.0, 0.0, -0.6}, 3.0, 135.0, -20.0, false);

    std::unique_ptr<mjData, void (*)(mjData*)> render_data(mj_makeData(sim_core.model()), mj_deleteData);
    if (!render_data) {
        std::cerr << "Failed to allocate MuJoCo render snapshot.\n";
        return EXIT_FAILURE;
    }

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> action_1_dist(-config.max_torques[0], config.max_torques[0]);
    std::uniform_real_distribution<double> action_2_dist(-config.max_torques[1], config.max_torques[1]);

    std::atomic<bool> run_simulation{true};
    const auto env_step_period = std::chrono::duration<double>(static_cast<double>(config.repeat_action) /
                                                               static_cast<double>(config.simulation_frequency));
    const auto render_period = std::chrono::duration<double>(1.0 / 60.0);

    std::thread simulation_thread([&]() {
        using clock_type = std::chrono::steady_clock;
        auto next_tick = clock_type::now();

        while (run_simulation.load(std::memory_order_acquire)) {
            next_tick += std::chrono::duration_cast<clock_type::duration>(env_step_period);

            if (reset_requested.exchange(false, std::memory_order_acq_rel)) {
                observation = env.reset();
            }

            if (!paused.load(std::memory_order_acquire)) {
                const std::vector<double> action = {action_1_dist(rng), action_2_dist(rng)};
                const auto result = env.step(action);
                observation = result.observations;

                if (result.terminated || result.truncated) {
                    observation = env.reset();
                }
            }

            std::this_thread::sleep_until(next_tick);
            const auto now = clock_type::now();
            if (now > next_tick + std::chrono::duration_cast<clock_type::duration>(env_step_period)) {
                next_tick = now;
            }
        }
    });

    while (!viewer.should_close()) {
        const auto frame_start = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::recursive_mutex> lock(sim_core.state_mutex());
            mj_copyData(render_data.get(), sim_core.model(), sim_core.data());
        }

        viewer.update_scene(*render_data);
        viewer.present();

        std::this_thread::sleep_until(frame_start + render_period);
    }

    run_simulation.store(false, std::memory_order_release);
    simulation_thread.join();

    return EXIT_SUCCESS;
}
