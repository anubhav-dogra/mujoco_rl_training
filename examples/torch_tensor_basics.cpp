#include <ATen/TensorIndexing.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/slice.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << torch::cuda::is_available() << std::endl;
    torch::Tensor x = torch::tensor({1.0, 2.0, 3.0}).to(device);  //.to(device);
    torch::Tensor y = torch::eye(3).to(device);
    std::cout << x << std::endl;  // prints out as coloum vector, but is it row?
    std::cout << y << std::endl;

    std::cout << x + y << std::endl;                // added x into each row of the matrix
    std::cout << x.sizes() << std::endl;            // [3]
    std::cout << x[0].item<double>() << std::endl;  // obviously first element
    torch::Tensor m = torch::arange(0, 6).reshape({2, 3});
    std::cout << m << std::endl;
    std::cout << m.sizes() << std::endl;  // [2, 3]
    std::cout << m.index({torch::indexing::Slice(), 1}) << std::endl;
    return 0;
}
