#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

inline torch::Device default_device() {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
}

}  // namespace mujoco_rl_training
