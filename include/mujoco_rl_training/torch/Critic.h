#pragma once

#include <torch/torch.h>

namespace mujoco_rl_training {

// -----------------------------------------------------------------------------
// TorchCritic: state-value baseline V(s) for PPO's GAE/advantage computation.
//
// Architecture mirrors the actor:
//   obs -> Linear(obs_dim, 64) -> tanh
//       -> Linear(64, 64)      -> tanh
//       -> Linear(64, 1)            = predicted V(s) (one scalar per state)
//
// Trained to regress on Monte-Carlo returns (GAE returns = adv + value) via
// MSE. The critic is NOT used during deployment — only during training, to
// reduce variance of policy-gradient estimates by subtracting V(s) from
// rewards-to-go (this is the "advantage" baseline trick).
// -----------------------------------------------------------------------------
struct TorchCriticImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear out{nullptr};

    explicit TorchCriticImpl(int obs_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        out = register_module("out", torch::nn::Linear(64, 1));
    }

    // Returns V(obs) with shape [batch, 1].
    torch::Tensor forward(torch::Tensor obs) {
        auto x = torch::tanh(fc1->forward(obs));
        x = torch::tanh(fc2->forward(x));
        return out->forward(x);
    }
};

TORCH_MODULE(TorchCritic);

}  // namespace mujoco_rl_training
