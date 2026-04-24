#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mujoco_rl_training {

inline std::ofstream open_artifact_output(const std::string& path) {
    std::filesystem::create_directories("artifacts");

    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open artifact for writing: " + path);
    }

    return output;
}

}  // namespace mujoco_rl_training
