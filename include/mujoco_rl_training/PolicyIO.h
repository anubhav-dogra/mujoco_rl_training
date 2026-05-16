#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace mujoco_rl_training {

inline void ensure_artifact_parent_directory(const std::string& path) {
    const auto parent_path = std::filesystem::path(path).parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }
}

inline std::ofstream open_artifact_output(const std::string& path) {
    ensure_artifact_parent_directory(path);

    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open artifact for writing: " + path);
    }

    return output;
}

inline std::ofstream open_artifact_append(const std::string& path) {
    ensure_artifact_parent_directory(path);

    std::ofstream output(path, std::ios::app);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open artifact for appending: " + path);
    }

    return output;
}

}  // namespace mujoco_rl_training
