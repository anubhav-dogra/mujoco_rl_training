#include <torch/torch.h>
#include <torch/utils.h>
int main() {
    auto w = torch::tensor({0.0}, torch::requires_grad());

    for (auto i = 0; i < 100; ++i) {
        auto prediction = w * 2.0;
        auto loss = torch::pow(prediction - 10.0, 2);

        loss.backward();
        {
            torch::NoGradGuard no_grad;
            w -= 0.01 * w.grad();
            w.grad().zero_();
        }
        if (i % 10 == 0) {
            std::cout << i << "loss " << loss.item<double>() << std::endl << "w = " << w.item<double>() << std::endl;
        }
    }
    return 0;
}
