#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor w = torch::tensor({2.0}, torch::requires_grad());
    auto loss = w * w + 3.0 * w;
    loss.backward();

    std::cout << "w = " << w << std::endl;
    std::cout << "loss = " << loss << std::endl;
    std::cout << "grad = " << w.grad() << std::endl;
    return 0;
}
