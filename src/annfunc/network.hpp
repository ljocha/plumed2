#include <cmath>
#include "./matrix.hpp"

template <typename Scalar>
class Network
{
using vector = std::vector<Scalar>;
using fn_t = Scalar (*)(Scalar);

private:
    int num_layers;
    double rescale_factor;

    std::vector<int> layer_sizes;

    std::vector<matrix<Scalar>> weights;
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
    , weights(std::vector<matrix<Scalar>>(num_layers - 1))
    , biases(std::vector<vector>(num_layers))
    , layers(std::vector<vector>(num_layers))
    , gradients(std::vector<vector>(num_layers-1))
    , activations(std::move(activations))
    , d_activations(std::move(d_activations))
    {
        for (int i = 0; i < num_layers - 1; ++i) {
            weights[i] = matrix<Scalar>(this->layer_sizes[i+1], this->layer_sizes[i]);
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

    void set_weights(int layer_index, const std::vector<double>& w) {
        std::copy(w.begin(), w.end(), weights[layer_index].data().begin());
    }

    void set_biases(int layer_index, const std::vector<double>& b) {
        std::copy(b.begin(), b.end(), biases[layer_index].begin());
    }

    vector& compute_output() {
        /* rescale the input */
        for (Scalar& v : layers[0] ) {
            v *= rescale_factor;
        }

        /* compute innier potentials in the hidden layer */
        for (int l = 1; l < num_layers; ++l) {
            std::copy(biases[l].begin(), biases[l].end(), layers[l].begin());
            weights[l-1].add_right_vector_multiply(layers[l-1], layers[l]);
            /* compute activations in the hidden layer */
            for (size_t i = 0; i < layers[l].size(); i++) {
                layers[l][i] = activations[l](layers[l][i]);
            }
        }

        return layers.back();
    }

    vector& compute_gradient(int output_index) {
        weights.back().copy_row(output_index, gradients.back());
        
        for (size_t i = 0; i < gradients.back().size(); ++i) {
            gradients.back()[i] *= d_activations[num_layers-1](layers[num_layers-1][output_index]) * d_activations[num_layers-2](layers[num_layers-2][i]);
        }

        for (int l = num_layers - 3; l >= 0; --l) {
            weights[l].left_vector_multiply(gradients[l+1], gradients[l]);
            
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

    static Scalar sigmoid(Scalar v) { return 1/(1 + exp(v)); }
    static Scalar d_sigmoid(Scalar fv) { return (1 - fv) * fv; }

    static Scalar linear(Scalar v) { return v; }
    static Scalar d_linear(Scalar) { return 1; }

    static Scalar relu(Scalar v) { return v < 0 ? 0 : v; }
    static Scalar d_relu(Scalar fv) { return fv <= 0 ? 0 : 1; }
};
