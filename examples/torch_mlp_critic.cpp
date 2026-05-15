#include "mujoco_rl_training/torch/Critic.h"
#include "mujoco_rl_training/torch/TensorUtils.h"

#include <torch/torch.h>

#include <iostream>

int main() {
    torch::manual_seed(0);

    const auto device = mujoco_rl_training::default_device();
    std::cout << "device: " << device << '\n';

    mujoco_rl_training::TorchCritic critic(6);
    critic->to(device);

    torch::optim::Adam optimizer(critic->parameters(), torch::optim::AdamOptions(0.01));

    auto obs = torch::randn({256, 6}).to(device);
    auto returns = obs.sum(1, true);

    for (int epoch = 0; epoch < 500; ++epoch) {
        auto values = critic->forward(obs);
        auto loss = torch::mse_loss(values, returns);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        if (epoch % 50 == 0) {
            std::cout << "epoch=" << epoch << " loss=" << loss.item<double>() << '\n';
        }
    }
    auto test_obs = torch::tensor({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}).to(device);
    auto predicted_value = critic->forward(test_obs);

    std::cout << "target value: " << test_obs.sum().item<double>() << '\n';
    std::cout << "predicted value: " << predicted_value.item<double>() << '\n';

    return 0;
}
