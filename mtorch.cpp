/*
//#include <torch/extension.h>
#include <torch/torch.h>

int main() {
	return 0;
}*/

#include <torch/torch.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.to(torch::kMPS);
    std::cout << tensor << std::endl;
    std::cout << tensor.data_ptr() << std::endl;

    return 0;
}

