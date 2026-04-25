#pragma once

#include <mujoco_rl_training/DoublePendulumEnv.h>
#include <mujoco_rl_training/PolicyIO.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mujoco_rl_training {

struct DoublePendulumPolicyMetadata {
    DoublePendulumEnvConfig config;
    double best_return = 0.0;
    int num_iterations = 0;
    int episodes_per_evaluation = 0;
    double noise_std_dev = 0.0;
};

inline bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string double_pendulum_metadata_path_for_policy(const std::string& policy_path) {
    if (ends_with(policy_path, ".txt")) {
        return policy_path.substr(0, policy_path.size() - 4) + ".meta.txt";
    }
    return policy_path + ".meta.txt";
}

namespace detail {

inline std::string trim(std::string value) {
    const auto is_not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), is_not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), is_not_space).base(), value.end());
    return value;
}

template <typename Values>
inline void write_vector(std::ostream& output, const std::string& key, const Values& values) {
    output << key << "=[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << ", ";
        }
        output << values[i];
    }
    output << "]\n";
}

inline std::string vector_body(const std::string& value, const std::string& key) {
    const std::string trimmed = trim(value);
    if (trimmed.size() < 2 || trimmed.front() != '[' || trimmed.back() != ']') {
        throw std::runtime_error("Expected vector value for metadata key: " + key);
    }
    return trimmed.substr(1, trimmed.size() - 2);
}

inline std::vector<std::string> parse_string_vector(const std::string& value, const std::string& key) {
    std::vector<std::string> values;
    std::stringstream stream(vector_body(value, key));
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::string trimmed = trim(item);
        if (!trimmed.empty()) {
            values.push_back(trimmed);
        }
    }
    return values;
}

inline std::vector<double> parse_double_vector(const std::string& value, const std::string& key) {
    std::vector<double> values;
    for (const auto& item : parse_string_vector(value, key)) {
        values.push_back(std::stod(item));
    }
    return values;
}

inline std::unordered_map<std::string, std::string> load_key_values(const std::string& metadata_path) {
    std::ifstream input(metadata_path);
    if (!input.is_open()) {
        return {};
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') {
            continue;
        }

        const auto separator = line.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Malformed metadata line in " + metadata_path + ": " + line);
        }
        metadata[trim(line.substr(0, separator))] = trim(line.substr(separator + 1));
    }
    return metadata;
}

}  // namespace detail

inline void save_double_pendulum_policy_metadata(const std::string& metadata_path, const std::string& policy_path,
                                                 const DoublePendulumEnvConfig& config, double best_return,
                                                 int num_iterations, int episodes_per_evaluation,
                                                 double noise_std_dev) {
    auto output = open_artifact_output(metadata_path);
    output << std::setprecision(17);

    output << "policy_artifact=" << policy_path << '\n';
    output << "best_return=" << best_return << '\n';
    output << "num_iterations=" << num_iterations << '\n';
    output << "episodes_per_evaluation=" << episodes_per_evaluation << '\n';
    output << "noise_std_dev=" << noise_std_dev << '\n';
    output << "xml_path=" << config.xml_path << '\n';
    output << "simulation_frequency=" << config.simulation_frequency << '\n';
    output << "episode_horizon=" << config.episode_horizon << '\n';
    output << "repeat_action=" << config.repeat_action << '\n';
    output << "seed=" << config.seed << '\n';
    output << "reset_angle_range=" << config.reset_angle_range << '\n';
    output << "reset_velocity_range=" << config.reset_velocity_range << '\n';
    detail::write_vector(output, "joint_names", config.joint_names);
    detail::write_vector(output, "target_angles", config.target_angles);
    detail::write_vector(output, "angle_cost_weights", config.angle_cost_weights);
    detail::write_vector(output, "velocity_cost_weights", config.velocity_cost_weights);
    detail::write_vector(output, "control_cost_weights", config.control_cost_weights);
    detail::write_vector(output, "max_torques", config.max_torques);
}

inline std::optional<DoublePendulumPolicyMetadata> load_double_pendulum_policy_metadata(
    const std::string& metadata_path) {
    const auto metadata = detail::load_key_values(metadata_path);
    if (metadata.empty()) {
        return std::nullopt;
    }

    const auto value = [&](const std::string& key) -> const std::string& {
        const auto it = metadata.find(key);
        if (it == metadata.end()) {
            throw std::runtime_error("Missing metadata key '" + key + "' in " + metadata_path);
        }
        return it->second;
    };

    DoublePendulumPolicyMetadata loaded;
    loaded.best_return = std::stod(value("best_return"));
    loaded.num_iterations = std::stoi(value("num_iterations"));
    loaded.episodes_per_evaluation = std::stoi(value("episodes_per_evaluation"));
    loaded.noise_std_dev = std::stod(value("noise_std_dev"));
    loaded.config.xml_path = value("xml_path");
    loaded.config.simulation_frequency = std::stoi(value("simulation_frequency"));
    loaded.config.episode_horizon = std::stoi(value("episode_horizon"));
    loaded.config.repeat_action = std::stoi(value("repeat_action"));
    loaded.config.seed = static_cast<unsigned int>(std::stoul(value("seed")));
    loaded.config.reset_angle_range = std::stod(value("reset_angle_range"));
    loaded.config.reset_velocity_range = std::stod(value("reset_velocity_range"));
    loaded.config.joint_names = detail::parse_string_vector(value("joint_names"), "joint_names");
    loaded.config.target_angles = detail::parse_double_vector(value("target_angles"), "target_angles");
    loaded.config.angle_cost_weights = detail::parse_double_vector(value("angle_cost_weights"), "angle_cost_weights");
    loaded.config.velocity_cost_weights =
        detail::parse_double_vector(value("velocity_cost_weights"), "velocity_cost_weights");
    loaded.config.control_cost_weights =
        detail::parse_double_vector(value("control_cost_weights"), "control_cost_weights");
    loaded.config.max_torques = detail::parse_double_vector(value("max_torques"), "max_torques");
    return loaded;
}

}  // namespace mujoco_rl_training
