#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

struct TorchActorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear mean{nullptr};

    explicit TorchActorImpl(int obs_dim, int action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        mean = register_module("mean", torch::nn::Linear(64, action_dim));
    }

    torch::Tensor forward(torch::Tensor obs) {
        auto x = torch::tanh(fc1->forward(obs));
        x = torch::tanh(fc2->forward(x));
        return mean->forward(x);
    }
};

TORCH_MODULE(TorchActor);

}  // namespace mujoco_rl_training
