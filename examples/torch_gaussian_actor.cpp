#include "mujoco_rl_training/torch/Actor.h"
#include "mujoco_rl_training/torch/GaussianPolicy.h"
#include "mujoco_rl_training/torch/TensorUtils.h"

#include <torch/torch.h>

#include <iostream>

int main() {
    torch::manual_seed(0);
    constexpr int obs_dim = 6;
    constexpr int action_dim = 2;

    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    mujoco_rl_training::TorchActor actor(obs_dim, action_dim);
    actor->to(device);

    auto obs = torch::randn({4, obs_dim}).to(device);

    auto mean_action = actor->forward(obs);

    auto log_std = torch::full({action_dim}, -0.5).to(device);
    auto std = torch::exp(log_std);

    auto raw_action = mujoco_rl_training::sample_gaussian_action(mean_action, log_std);
    auto normalized_action = mujoco_rl_training::squash_action(raw_action);
    std::cout << "obs shape: " << obs.sizes() << '\n';
    std::cout << "mean action:\n" << mean_action.cpu() << '\n';
    std::cout << "std:\n" << std.cpu() << '\n';
    std::cout << "raw sampled action:\n" << raw_action.cpu() << '\n';
    std::cout << "normalized action:\n" << normalized_action.cpu() << '\n';

    return 0;
}
