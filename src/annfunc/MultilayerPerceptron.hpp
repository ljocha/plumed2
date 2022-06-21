#include <cmath>
#include "tools/Matrix.h"

namespace PLMD {

template <typename Scalar>
class MultilayerPerceptron
{
using matrix = PLMD::Matrix<Scalar>;
using vector = std::vector<Scalar>;
using fn_t = Scalar (*)(Scalar);

private:
    int numLayers;
    double rescaleFactor;

    std::vector<int> layerSizes;

    std::vector<matrix> weights;
    std::vector<vector> biases;
    
    std::vector<vector> layers;
    std::vector<vector> gradients;

    std::vector<fn_t> activations;
    std::vector<fn_t> d_activations;

public:
    explicit MultilayerPerceptron(std::vector<int> layerSizes, std::vector<fn_t> activations, std::vector<fn_t> d_activations, Scalar rescaleFactor = 1)
    : numLayers(layerSizes.size())
    , rescaleFactor(rescaleFactor)
    , layerSizes(std::move(layerSizes))
    , weights(std::vector<matrix>(numLayers - 1))
    , biases(std::vector<vector>(numLayers))
    , layers(std::vector<vector>(numLayers))
    , gradients(std::vector<vector>(numLayers-1))
    , activations(std::move(activations))
    , d_activations(std::move(d_activations))
    {
        for (int i = 0; i < numLayers - 1; ++i) {
            weights[i] = matrix(this->layerSizes[i+1], this->layerSizes[i]);
        }

        for (int i = 0; i < numLayers; ++i) {
            biases[i] = vector(this->layerSizes[i], 0);
        }

        for (int i = 0; i < numLayers; ++i) {
            layers[i] = vector(this->layerSizes[i]);
        }

        for (int i = 0; i < numLayers-1; ++i) {
            gradients[i] = vector(this->layerSizes[i]);
        }
    }

    void setWeights(int layerIndex, const std::vector<Scalar>& w) {
        weights[layerIndex].setFromVector(w);
    }

    void setBiases(int layerIndex, const std::vector<Scalar>& b) {
        std::copy(b.begin(), b.end(), biases[layerIndex].begin());
    }

    int outputSize() const {
        return layers.back().size();
    }

    const vector& computeOutput() {
        /* rescale the input */
        for (Scalar& v : layers[0] ) {
            v *= rescaleFactor;
        }

        /* compute innier potentials in the hidden layer */
        for (int l = 1; l < numLayers; ++l) {
            mult(weights[l-1], layers[l-1], layers[l]);
            for (uint i = 0; i < biases[l].size(); ++i) layers[l][i] += biases[l][i];

            /* compute activations in the hidden layer */
            for (size_t i = 0; i < layers[l].size(); i++) {
                layers[l][i] = activations[l](layers[l][i]);
            }
        }

        return layers.back();
    }

    const vector& computeGradient(int outputIndex) {
        for (uint j = 0; j < gradients.back().size(); ++j) {
            gradients.back()[j] = weights.back()(outputIndex, j);
        }
        
        for (size_t i = 0; i < gradients.back().size(); ++i) {
            gradients.back()[i] *= d_activations[numLayers-1](layers[numLayers-1][outputIndex]) * d_activations[numLayers-2](layers[numLayers-2][i]);
        }

        for (int l = numLayers - 3; l >= 0; --l) {
            mult(gradients[l+1], weights[l], gradients[l]);

            for (size_t i = 0; i < gradients[l].size(); ++i) {
                gradients[l][i] *= d_activations[l](layers[l][i]);
            }
        }

        /* rescale the input gradient */
        for (Scalar& g : gradients.front()) {
            g *= rescaleFactor;
        }

        return gradients.front();
    }

    vector& getInput() {
        return layers.front();
    }

    /*
    * Derivatives are computed in terms of the output value f(v).
    */
    static Scalar tanh(Scalar v) { return std::tanh(v); }
    static Scalar d_tanh(Scalar fv) { return 1 - fv*fv; }

    static Scalar sigmoid(Scalar v) { return 1/(1 + exp(-v)); }
    static Scalar d_sigmoid(Scalar fv) { return (1 - fv) * fv; }

    static Scalar linear(Scalar v) { return v; }
    static Scalar d_linear(Scalar) { return 1; }

    static Scalar relu(Scalar v) { return v < 0 ? 0 : v; }
    static Scalar d_relu(Scalar fv) { return fv <= 0 ? 0 : 1; }
};

} // namespace PLMD
