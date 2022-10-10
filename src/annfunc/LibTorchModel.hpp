#include <torch/script.h>
#include <torch/csrc/api/include/torch/autograd.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace PLMD {

class LibTorchModel
{
private:
    using Module = torch::jit::script::Module;

    int inputSize;
    int _outputSize;

    torch::Tensor input;
    torch::Tensor output;
    Module model;

public:
    explicit LibTorchModel(std::string model_file_name, int inputSize)
    : inputSize(inputSize)
    , input(torch::empty({inputSize}))
    , model(torch::jit::load(model_file_name))
    {
        // Forward propagation on dummy input
        input.requires_grad_(true);
        output = model({input}).toTensor();

        // Output dimension check
        if (output.dim() > 1) {
            throw std::invalid_argument("Wrong output shape. Expected a 0D or 1D tensor.");
        }

        if (output.dim() == 0) {
            output = output.reshape(1);
        }
        
        _outputSize = output.sizes()[0];

        // Backward propagation check
        auto v = torch::zeros_like(output);
        auto gradient = torch::autograd::grad({output}, {input}, {v}, true)[0];
        if (gradient.sizes()[0] != inputSize) {
            throw std::invalid_argument("Wrong input shape. Expected a vector of length " + std::to_string(inputSize));
        }
    }

    int outputSize() const {
        return _outputSize;
    }

    torch::Tensor computeOutput() {
        input.requires_grad_(true);
        output = model({input}).toTensor();
        if (output.dim() == 0) output = output.reshape(1);
        return output;
    }

    torch::Tensor computeGradient(int outputIndex) {
        auto v = torch::zeros_like(output);
        v[outputIndex] = 1;
        return torch::autograd::grad({output}, {input}, {v}, true)[0];
    }

    torch::Tensor& getInput() {
        input.requires_grad_(false);
        return input;
    }
};

} // namespace PLMD
