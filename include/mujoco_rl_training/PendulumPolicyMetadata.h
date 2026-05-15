#pragma once

#include <envs/PendulumEnv.h>
#include <mujoco_rl_training/PolicyIO.h>

#include <fstream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mujoco_rl_training {

enum class PendulumPolicyActionScale {
    PhysicalTorque,
    NormalizedTorque,
};

struct PendulumPolicyMetadata {
    PendulumEnvConfig config;
    PendulumPolicyActionScale action_scale = PendulumPolicyActionScale::PhysicalTorque;
    std::string algorithm{};
    std::optional<double> best_return{};
};

inline std::string pendulum_metadata_path_for_policy(const std::string& policy_path) {
    constexpr const char* kTxtSuffix = ".txt";
    if (policy_path.size() >= 4 && policy_path.compare(policy_path.size() - 4, 4, kTxtSuffix) == 0) {
        return policy_path.substr(0, policy_path.size() - 4) + ".meta.txt";
    }
    return policy_path + ".meta.txt";
}

inline const char* to_string(PendulumPolicyActionScale action_scale) {
    switch (action_scale) {
        case PendulumPolicyActionScale::PhysicalTorque:
            return "physical_torque";
        case PendulumPolicyActionScale::NormalizedTorque:
            return "normalized_torque";
    }
    return "physical_torque";
}

inline PendulumPolicyActionScale parse_pendulum_action_scale(const std::string& value) {
    if (value == "physical_torque") {
        return PendulumPolicyActionScale::PhysicalTorque;
    }
    if (value == "normalized_torque") {
        return PendulumPolicyActionScale::NormalizedTorque;
    }
    throw std::runtime_error("Unknown pendulum policy action_scale: " + value);
}

inline double pendulum_physical_action(double policy_action, PendulumPolicyActionScale action_scale, double max_torque) {
    if (action_scale == PendulumPolicyActionScale::NormalizedTorque) {
        return max_torque * policy_action;
    }
    return policy_action;
}

inline void save_pendulum_policy_metadata(const std::string& metadata_path, const std::string& policy_path,
                                          const PendulumEnvConfig& config,
                                          PendulumPolicyActionScale action_scale,
                                          const std::string& algorithm,
                                          const std::optional<double>& best_return = std::nullopt) {
    auto output = open_artifact_output(metadata_path);
    output << std::setprecision(17);
    output << "policy_artifact=" << policy_path << '\n';
    output << "algorithm=" << algorithm << '\n';
    output << "action_scale=" << to_string(action_scale) << '\n';
    if (best_return.has_value()) {
        output << "best_return=" << best_return.value() << '\n';
    }
    output << "xml_path=" << config.xml_path << '\n';
    output << "simulation_frequency=" << config.simulation_frequency << '\n';
    output << "max_torque=" << config.max_torque << '\n';
    output << "episode_horizon=" << config.episode_horizon << '\n';
    output << "seed=" << config.seed << '\n';
    output << "repeat_action=" << config.repeat_action << '\n';
    output << "reset_angle_range=" << config.reset_angle_range << '\n';
    output << "reset_velocity_range=" << config.reset_velocity_range << '\n';
}

inline std::optional<PendulumPolicyMetadata> load_pendulum_policy_metadata(const std::string& metadata_path) {
    std::ifstream input(metadata_path);
    if (!input.is_open()) {
        return std::nullopt;
    }

    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Malformed pendulum policy metadata line: " + line);
        }
        values[line.substr(0, separator)] = line.substr(separator + 1);
    }

    const auto value = [&](const std::string& key) -> const std::string& {
        const auto it = values.find(key);
        if (it == values.end()) {
            throw std::runtime_error("Missing pendulum policy metadata key: " + key);
        }
        return it->second;
    };

    PendulumPolicyMetadata metadata;
    metadata.algorithm = value("algorithm");
    metadata.action_scale = parse_pendulum_action_scale(value("action_scale"));
    metadata.config.xml_path = value("xml_path");
    metadata.config.simulation_frequency = std::stoi(value("simulation_frequency"));
    metadata.config.max_torque = std::stod(value("max_torque"));
    metadata.config.episode_horizon = std::stoi(value("episode_horizon"));
    metadata.config.seed = static_cast<unsigned int>(std::stoul(value("seed")));
    metadata.config.repeat_action = std::stoi(value("repeat_action"));
    metadata.config.reset_angle_range = std::stod(value("reset_angle_range"));
    metadata.config.reset_velocity_range = std::stod(value("reset_velocity_range"));

    const auto best_return_it = values.find("best_return");
    if (best_return_it != values.end()) {
        metadata.best_return = std::stod(best_return_it->second);
    }
    return metadata;
}

}  // namespace mujoco_rl_training
