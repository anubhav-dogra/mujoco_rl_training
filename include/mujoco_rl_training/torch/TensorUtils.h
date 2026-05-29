#pragma once
#include <cstddef>
#include <cstdint>
#include <mujoco_rl_training/ActionUtils.hpp>
#include <vector>
#include <stdexcept>

#include <torch/torch.h>

namespace mujoco_rl_training {

// =============================================================================
// Tensor / std::vector bridge helpers.
//
// The env layer speaks `std::vector<double>` (CPU, double precision). The
// network layer speaks `torch::Tensor` (often GPU, float32). These helpers
// move data across the boundary efficiently, with the necessary `.clone()`
// where a stack-allocated source vector would otherwise dangle.
// =============================================================================

// CUDA when available, CPU otherwise. Called once per program to pick where
// the networks live.
inline torch::Device default_device() {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
}

// Tensor (any shape, any device, any dtype) -> flat std::vector<double>.
// We force to CPU + float64 + contiguous so .item<double>() is safe per cell.
inline std::vector<double> tensor_to_vector(const torch::Tensor& tensor) {
    const auto cpu_tensor = tensor.detach().to(torch::kCPU).to(torch::kFloat64).contiguous().view({-1});
    std::vector<double> vector;
    vector.reserve(static_cast<std::size_t>(cpu_tensor.size(0)));
    for (int64_t i = 0; i < cpu_tensor.size(0); ++i) {
        vector.push_back(cpu_tensor.index({i}).item<double>());
    }
    return vector;
}

// std::vector<double> -> torch::Tensor on `device`, float32.
//
// `from_blob` shares storage with the source buffer, so we IMMEDIATELY clone
// to detach. Without the clone, the tensor would dangle once `flat` goes out
// of scope. `add_batch_dimension=true` unsqueezes to shape [1, N] for
// per-sample forward calls.
inline torch::Tensor vector_to_tensor(const std::vector<double>& values, const torch::Device& device,
                                      bool add_batch_dimension = false) {
    std::vector<float> flat;
    flat.reserve(values.size());
    for (const double& value : values) {
        flat.push_back(static_cast<float>(value));
    }
    auto tensor = torch::from_blob(flat.data(), {static_cast<int64_t>(flat.size())},
                                   torch::TensorOptions().dtype(torch::kFloat32))
                      .clone()
                      .to(device);
    if (add_batch_dimension) {
        tensor = tensor.unsqueeze(0);
    }
    return tensor;
}

// std::vector<double> -> [N, 1] column tensor, useful for shapes that need to
// broadcast against [N, action_dim] tensors during PPO loss math.
inline torch::Tensor vector_to_column_tensor(const std::vector<double>& values, const torch::Device& device) {
    return vector_to_tensor(values, device).unsqueeze(1);
}

// std::vector<std::vector<double>> -> [rows, cols] tensor on `device`, float32.
// Rejects ragged inputs (each row must be the same length).
inline torch::Tensor matrix_to_tensor(const std::vector<std::vector<double>>& values, const torch::Device& device) {
    if (values.empty()) {
        throw std::runtime_error("matrix_to_tensor: cannot convert an empty matrix");
    }

    const std::size_t rows = values.size();
    const std::size_t cols = values.front().size();
    std::vector<float> flat;
    flat.reserve(rows * cols);

    for (const auto& row : values) {
        if (row.size() != cols) {
            throw std::runtime_error("matrix_to_tensor: ragged input matrix");
        }
        for (double value : row) {
            flat.push_back(static_cast<float>(value));
        }
    }

    // Clone for the same dangling-storage reason as vector_to_tensor.
    return torch::from_blob(flat.data(), {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

// Convenience: Tensor -> scaled-action vector in one shot (symmetric limits).
inline std::vector<double> scale_action(const torch::Tensor& normalized_actions,
                                        const std::vector<double>& action_limits) {
    return mujoco_rl_training::scale_action(mujoco_rl_training::tensor_to_vector(normalized_actions), action_limits);
}
}  // namespace mujoco_rl_training
