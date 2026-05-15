#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

struct TorchCriticImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear out{nullptr};

    explicit TorchCriticImpl(int obs_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        out = register_module("out", torch::nn::Linear(64, 1));
    }

    torch::Tensor forward(torch::Tensor obs) {
        auto x = torch::tanh(fc1->forward(obs));
        x = torch::tanh(fc2->forward(x));
        return out->forward(x);
    }
};

TORCH_MODULE(TorchCritic);

}  // namespace mujoco_rl_training
