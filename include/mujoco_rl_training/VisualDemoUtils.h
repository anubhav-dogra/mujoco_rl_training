#pragma once

#include <mujoco/mujoco.h>
#include <mujoco_viewer/MujocoViewer.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace mujoco_rl_training {

struct ViewerCameraConfig {
    std::array<double, 3> lookat{};
    double distance = 2.5;
    double azimuth = 135.0;
    double elevation = -30.0;
};

template <typename Env, typename Observation, typename ActionSampler>
int run_visual_demo(Env& env, Observation& observation, ActionSampler sample_action, const ViewerCameraConfig& camera) {
    auto& sim_core = env.sim_core();
    std::atomic<bool> paused{false};
    std::atomic<bool> reset_requested{false};

    MujocoViewer viewer;
    viewer.initialize(*sim_core.model(), false, &paused, &reset_requested);
    viewer.set_camera_properties({camera.lookat[0], camera.lookat[1], camera.lookat[2]}, camera.distance,
                                 camera.azimuth, camera.elevation, false);

    std::unique_ptr<mjData, void (*)(mjData*)> render_data(mj_makeData(sim_core.model()), mj_deleteData);
    if (!render_data) {
        std::cerr << "Failed to allocate MuJoCo render snapshot.\n";
        return EXIT_FAILURE;
    }

    std::atomic<bool> run_simulation{true};
    const auto env_step_period = std::chrono::duration<double>(static_cast<double>(env.config().repeat_action) /
                                                               static_cast<double>(env.config().simulation_frequency));
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
                const auto action = sample_action();
                const auto result = env.step(action);
                observation = result.observation;

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

}  // namespace mujoco_rl_training
