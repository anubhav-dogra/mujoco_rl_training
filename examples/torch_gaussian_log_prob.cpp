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
    torch::optim::Adam optimizer(actor->parameters(), torch::optim::AdamOptions(0.01));

    auto obs = torch::randn({2, obs_dim}).to(device);
    auto mean_before = actor->forward(obs).detach();

    auto mean = actor->forward(obs);
    auto log_std = torch::full({action_dim}, -0.5).to(device);
    auto std = torch::exp(log_std);
    auto action = mujoco_rl_training::sample_gaussian_action(mean, log_std).detach();

    auto log_prob = mujoco_rl_training::gaussian_log_prob(action, mean, log_std);
    auto advantages = torch::tensor({{1.0}, {-1.0}}).to(device);
    auto actor_loss = -(log_prob * advantages).mean();

    optimizer.zero_grad();
    actor_loss.backward();
    optimizer.step();

    auto mean_after = actor->forward(obs).detach();

    std::cout << "mean before update:\n" << mean_before.cpu() << '\n';
    std::cout << "std:\n" << std.cpu() << '\n';
    std::cout << "sampled action:\n" << action.cpu() << '\n';
    std::cout << "log_prob:\n" << log_prob.cpu() << '\n';
    std::cout << "advantages:\n" << advantages.cpu() << '\n';
    std::cout << "actor_loss:\n" << actor_loss.cpu() << '\n';
    std::cout << "mean after update:\n" << mean_after.cpu() << '\n';
    return 0;
}
