#include <cmath>
// #include "./matrix.hpp"
#include "../tools/Matrix.h"

namespace PLMD {

template <typename Scalar>
class Network
{
using matrix = PLMD::Matrix<Scalar>;
using vector = std::vector<Scalar>;
using fn_t = Scalar (*)(Scalar);

private:
    int num_layers;
    double rescale_factor;

    std::vector<int> layer_sizes;

    std::vector<matrix> weights;
    std::vector<vector> biases;
    
    std::vector<vector> layers;
    std::vector<vector> gradients;

    std::vector<fn_t> activations;
    std::vector<fn_t> d_activations;

public:
    explicit Network(std::vector<int> layer_sizes, std::vector<fn_t> activations, std::vector<fn_t> d_activations, Scalar rescale_factor = 1)
    : num_layers(layer_sizes.size())
    , rescale_factor(rescale_factor)
    , layer_sizes(std::move(layer_sizes))
    , weights(std::vector<matrix>(num_layers - 1))
    , biases(std::vector<vector>(num_layers))
    , layers(std::vector<vector>(num_layers))
    , gradients(std::vector<vector>(num_layers-1))
    , activations(std::move(activations))
    , d_activations(std::move(d_activations))
    {
        for (int i = 0; i < num_layers - 1; ++i) {
            weights[i] = matrix(this->layer_sizes[i+1], this->layer_sizes[i]);
        }

        for (int i = 0; i < num_layers; ++i) {
            biases[i] = vector(this->layer_sizes[i], 0);
        }

        for (int i = 0; i < num_layers; ++i) {
            layers[i] = vector(this->layer_sizes[i]);
        }

        for (int i = 0; i < num_layers-1; ++i) {
            gradients[i] = vector(this->layer_sizes[i]);
        }
    }

    void set_weights(int layer_index, const std::vector<Scalar>& w) {
        weights[layer_index].setFromVector(w);
    }

    void set_biases(int layer_index, const std::vector<Scalar>& b) {
        std::copy(b.begin(), b.end(), biases[layer_index].begin());
    }

    vector& compute_output() {
        /* rescale the input */
        for (Scalar& v : layers[0] ) {
            v *= rescale_factor;
        }

        /* compute innier potentials in the hidden layer */
        for (int l = 1; l < num_layers; ++l) {
            mult(weights[l-1], layers[l-1], layers[l]);
            for (uint i = 0; i < biases[l].size(); ++i) layers[l][i] += biases[l][i];

            /* compute activations in the hidden layer */
            for (size_t i = 0; i < layers[l].size(); i++) {
                layers[l][i] = activations[l](layers[l][i]);
            }
        }

        return layers.back();
    }

    vector& compute_gradient(int output_index) {
        for (uint j = 0; j < gradients.back().size(); ++j) {
            gradients.back()[j] = weights.back()(output_index, j);
        }
        
        for (size_t i = 0; i < gradients.back().size(); ++i) {
            gradients.back()[i] *= d_activations[num_layers-1](layers[num_layers-1][output_index]) * d_activations[num_layers-2](layers[num_layers-2][i]);
        }

        for (int l = num_layers - 3; l >= 0; --l) {
            mult(gradients[l+1], weights[l], gradients[l]);

            for (size_t i = 0; i < gradients[l].size(); ++i) {
                gradients[l][i] *= d_activations[l](layers[l][i]);
            }
        }

        /* rescale the input gradient */
        for (Scalar& g : gradients.front()) {
            g *= rescale_factor;
        }

        return gradients.front();
    }

    vector& get_input() {
        return layers.front();
    }

    static Scalar tanh(Scalar v) {return std::tanh(v); }
    static Scalar d_tanh(Scalar fv) { return 1 - fv*fv; }

    static Scalar sigmoid(Scalar v) { return 1/(1 + exp(-v)); }
    static Scalar d_sigmoid(Scalar fv) { return (1 - fv) * fv; }

    static Scalar linear(Scalar v) { return v; }
    static Scalar d_linear(Scalar) { return 1; }

    static Scalar relu(Scalar v) { return v < 0 ? 0 : v; }
    static Scalar d_relu(Scalar fv) { return fv <= 0 ? 0 : 1; }
};

} // namespace PLMD
