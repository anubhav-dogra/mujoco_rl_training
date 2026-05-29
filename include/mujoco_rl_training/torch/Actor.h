#pragma once

#include <cmath>

#include <torch/torch.h>

namespace mujoco_rl_training {

// -----------------------------------------------------------------------------
// TorchActor: stochastic policy for continuous-action PPO.
//
// Architecture:
//   obs -> Linear(obs_dim, 64) -> tanh
//       -> Linear(64, 64)      -> tanh
//       -> Linear(64, action_dim)   = "mean" (pre-tanh action mean per dim)
//
// Exploration noise:
//   We sample actions from a diagonal Gaussian centered at `mean` with
//   per-dimension standard deviation exp(log_std). `log_std` is stored as a
//   registered parameter so Adam updates it during PPO (no manual schedule).
//   Initialized from `initial_std` (default 0.7). Output of the sampler is
//   later squashed via tanh and affinely mapped to the env's action range.
//
// Why log_std (not std directly)?
//   It keeps std strictly positive without any constraint trick, and gradients
//   wrt log_std are well-scaled (additive shifts on log_std correspond to
//   multiplicative changes in std).
//
// Why `register_parameter` instead of a free Tensor?
//   `actor->parameters()` then automatically includes log_std, so the Adam
//   optimizer touches it. It also gets saved/loaded with `torch::save(actor)`.
// -----------------------------------------------------------------------------
struct TorchActorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear mean{nullptr};
    // Per-action-dim log-standard-deviation of the Gaussian policy. Learnable.
    torch::Tensor log_std;

    explicit TorchActorImpl(int obs_dim, int action_dim, double initial_std = 0.7) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        mean = register_module("mean", torch::nn::Linear(64, action_dim));
        // Initial std=0.7 corresponds to log_std ~= -0.357. Generous initial noise
        // for exploration; PPO will move log_std as needed (and the entropy bonus
        // resists premature collapse).
        log_std = register_parameter("log_std", torch::full({action_dim}, std::log(initial_std)));
    }

    // forward returns the pre-squash Gaussian mean. The training loop turns this
    // into an executed action via tanh(sample) and affine scaling.
    torch::Tensor forward(torch::Tensor obs) {
        auto x = torch::tanh(fc1->forward(obs));
        x = torch::tanh(fc2->forward(x));
        return mean->forward(x);
    }
};

// Wraps TorchActorImpl in a shared_ptr-like handle (`TorchActor actor; actor->...`).
TORCH_MODULE(TorchActor);

}  // namespace mujoco_rl_training
