#include <torch/torch.h>

#include <iostream>

int main() {
    torch::manual_seed(0);

    auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    torch::nn::Linear model(1, 1);
    model->to(device);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.05));

    auto x = torch::linspace(-1.0, 1.0, 100).reshape({100, 1}).to(device);
    auto y = 2.0 * x + 1.0;

    for (int epoch = 0; epoch < 200; ++epoch) {
        auto prediction = model->forward(x);
        auto loss = torch::mse_loss(prediction, y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 20 == 0) {
            std::cout << "epoch=" << epoch << " loss=" << loss.item<double>() << '\n';
        }
    }

    std::cout << "weight:\n" << model->weight.cpu() << '\n';
    std::cout << "bias:\n" << model->bias.cpu() << '\n';

    return 0;
}
