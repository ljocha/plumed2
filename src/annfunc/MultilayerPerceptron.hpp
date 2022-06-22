#include "tools/Matrix.h"

#include <algorithm>
#include <cmath>

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
        for (int l = 0; l < numLayers - 1; ++l) {
            weights[l] = matrix(this->layerSizes[l+1], this->layerSizes[l]);
        }

        for (int l = 0; l < numLayers; ++l) {
            biases[l] = vector(this->layerSizes[l]);
        }

        for (int l = 0; l < numLayers; ++l) {
            layers[l] = vector(this->layerSizes[l]);
        }

        for (int l = 0; l < numLayers - 1; ++l) {
            gradients[l] = vector(this->layerSizes[l]);
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
        std::transform(layers[0].begin(), layers[0].end(), layers[0].begin(), [&](Scalar v) { return v * rescaleFactor; });

        /* compute innier potentials in the hidden layer */
        for (int l = 1; l < numLayers; ++l) {
            mult(weights[l-1], layers[l-1], layers[l]);
            std::transform(
                layers[l].begin(), layers[l].end(),
                biases[l].begin(),
                layers[l].begin(),
                [&](Scalar v, Scalar b) { return v + b; }
            );

            /* compute activations in the hidden layer */
            std::transform(layers[l].begin(), layers[l].end(), layers[l].begin(), [&](Scalar v) { return activations[l](v); });
        }

        return layers.back();
    }

    const vector& computeGradient(int outputIndex) {
        for (uint i = 0; i < gradients.back().size(); ++i) {
            gradients.back()[i] = weights.back()(outputIndex, i);
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
        std::transform(gradients.front().begin(), gradients.front().end(), gradients.front().begin(), [&](Scalar v) { return v * rescaleFactor; });

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

    static Scalar sigmoid(Scalar v) { return 1/(1 + std::exp(-v)); }
    static Scalar d_sigmoid(Scalar fv) { return (1 - fv) * fv; }

    static Scalar linear(Scalar v) { return v; }
    static Scalar d_linear(Scalar) { return 1; }

    static Scalar relu(Scalar v) { return v < 0 ? 0 : v; }
    static Scalar d_relu(Scalar fv) { return fv <= 0 ? 0 : 1; }
};

} // namespace PLMD
