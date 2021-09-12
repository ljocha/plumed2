#include "../colvar/Colvar.h"
#include "../colvar/ActionRegister.h"

#include <cassert>
#include <string>
#include <cmath>
#include <iostream>
#include <memory>

#include "./network.hpp"

using namespace std;

// #define DEBUG
// #define DEBUG_2
// #define DEBUG_3

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR CMLP
/*
Multilayer perceptron colvar

Computes the function of multilayer perceptron with positions of atoms as input and a single output.
Optionally, the colvar can rescale the input by a specified constant.
The computation can be done with float or double precision.

\plumedfile
CMLP ...
LABEL=ann_cv1
ATOMS=1,4,12
RESCALE=0.100041
SIZES=3,2,1
ACTIVATIONS=TANH,TANH
WEIGHTS0=1.35345,2.353535,-3.242425,4.23424,5.25425,6.252343
WEIGHTS1=0.125232,0.224323
BIASES0=-0.005190,0.001307
BIASES1=0.213432
... CMLP
\endplumedfile

*/
//+ENDPLUMEDOC

class ColvarMLP : public Colvar
{
private:
    bool useDouble;
    std::vector<AtomNumber> atoms;
    std::unique_ptr<Network<double>> dNet;
    std::unique_ptr<Network<float>> fNet;

    std::vector<AtomNumber> parseAtoms();
    bool readFlag(std::string flagName);
    template <typename Scalar>
    std::unique_ptr<Network<Scalar>> parseNetwork();

    template <typename Scalar>
    void calculateByPrecision(Network<Scalar>& net);
public:
    static void registerKeywords(Keywords& keys);
    ColvarMLP(const ActionOptions& ao);
    virtual void calculate();
};

PLUMED_REGISTER_ACTION(ColvarMLP,"CMLP")

void ColvarMLP::registerKeywords( Keywords& keys ) {
    Colvar::registerKeywords(keys);

    keys.add("atoms","ATOMS","a list of atoms whose coordinates are input of the layer");
    keys.add("optional", "RESCALE", "a coefficient the input coordinates are multiplied by");

    keys.addFlag("DOUBLE_PREC", false, "whether to use double precision or not");
    keys.add("compulsory", "ACTIVATIONS", "array of activation functions in individual layers, F_1, ..., F_N; options TANH, LINEAR, RELU, SIGMOID");
    keys.add("compulsory", "SIZES", "array of sizes of individual layers, S_0, ..., S_n (S_0 should equal to three times the number of atoms)");
    keys.add("numbered", "WEIGHTS", "array of flattened weight matrices W_1, ..., W_N; matrix W_L of size S_{L-1} x S_L connects layers L-1 and L");
    keys.add("numbered", "BIASES", "array of bias vectors B_1, ..., B_N; vector B_L is added to the inner potential of layer L");

    keys.addOutputComponent("node", "default", "components of MLP outputs");
}

std::vector<AtomNumber> ColvarMLP::parseAtoms() {
    std::vector<AtomNumber> atoms;
    parseAtomList("ATOMS",atoms);
    return atoms;
}

bool ColvarMLP::readFlag(std::string flagName) {
    bool flag = false;
    parseFlag("DOUBLE_PREC", flag);
    return flag;
}

template <typename Scalar>
std::unique_ptr<Network<Scalar>> ColvarMLP::parseNetwork() {
    /* sizes of all the layers including the input layer */
    std::vector<int> layerSizes;
    parseVector("SIZES", layerSizes);
    
    int numLayers = layerSizes.size();

    std::vector<std::string> activationNames;
    std::vector<Scalar (*)(Scalar)> fns{Network<Scalar>::linear};
    std::vector<Scalar (*)(Scalar)> d_fns{Network<Scalar>::d_linear};

    /* activations in all the layers except for the input layer */
    parseVector("ACTIVATIONS", activationNames);
    for (auto& name : activationNames) {
        if (name == "TANH") {
            fns.push_back(Network<Scalar>::tanh);
            d_fns.push_back(Network<Scalar>::d_tanh);
        } else if (name == "LINEAR") {
            fns.push_back(Network<Scalar>::linear);
            d_fns.push_back(Network<Scalar>::d_linear);
        } else if (name == "SIGMOID") {
            fns.push_back(Network<Scalar>::sigmoid);
            d_fns.push_back(Network<Scalar>::d_sigmoid);
        } else if (name == "RELU") {
            fns.push_back(Network<Scalar>::relu);
            d_fns.push_back(Network<Scalar>::d_relu);
        } else {
            error("Unknown activation: " + name + ".");
        }
    }
    

    if (static_cast<int>(activationNames.size()) != numLayers - 1) error("Wrong number of activation functions given.");

    Scalar rescaleFactor = 1;
    std::unique_ptr<Network<Scalar>> dNet;
    parse("RESCALE", rescaleFactor);

    std::unique_ptr<Network<Scalar>> net(new Network<Scalar>(layerSizes, fns, d_fns, rescaleFactor));

    std::vector<Scalar> buffer;
    for (int l = 0; l < numLayers-1; ++l) {
        if(!parseNumberedVector("WEIGHTS", l, buffer)) error("Not enough weight matrices provided.");
        if (static_cast<int>(buffer.size()) != layerSizes[l] * layerSizes[l+1]) error("Invalid number of weights between layers " + to_string(l) + " and " + to_string(l+1) + ".");
        net->setWeights(l, buffer);
    }

    for (int l = 0; l < numLayers-1; ++l) {
        if(!parseNumberedVector("BIASES", l, buffer)) error("Not enough bias vectors provided.");
        if (static_cast<int>(buffer.size()) != layerSizes[l+1]) error("Invalid number of biases in layer " + to_string(l+1) + ".");
        net->setBiases(l+1, buffer);
    }

    return net;
}

ColvarMLP::ColvarMLP(const ActionOptions& ao)
    : PLUMED_COLVAR_INIT(ao)
    , useDouble(readFlag("DOUBLE_PREC"))
    , atoms(parseAtoms())
    , dNet(useDouble ? parseNetwork<double>() : nullptr)
    , fNet(useDouble ? nullptr : parseNetwork<float>())
{
    int outputSize = useDouble ? dNet->outputSize() : fNet->outputSize();

    for (int i = 0; i < outputSize; ++i) {
        addComponentWithDerivatives("node-" + to_string(i));
    }

    requestAtoms(atoms);

    checkRead();

    getPntrToValue()->setNotPeriodic();
}

void ColvarMLP::calculate() {
    if (useDouble) calculateByPrecision<double>(*dNet);
    else calculateByPrecision<float>(*fNet);
}

template<typename Scalar>
void ColvarMLP::calculateByPrecision(Network<Scalar>& net) {
    vector<Scalar>& input = net.getInput();

    vector<Vector> positions = getPositions();
    for (unsigned int i = 0; i < positions.size(); i++) {
        input[3 * i + 0] = positions[i][0];
        input[3 * i + 1] = positions[i][1];
        input[3 * i + 2] = positions[i][2];
    }

    const vector<Scalar>& output = net.computeOutput();
    for (uint i = 0; i < output.size(); ++i) {
        std::string compName = "node-" + to_string(i);
        Value* comp = getPntrToComponent(compName);
     
        comp->set(output[i]);
        const vector<Scalar>& derivatives = net.computeGradient(i);

        for (unsigned int i = 0; i < positions.size(); i++) {
            setAtomsDerivatives(comp, i, { derivatives[3 * i], derivatives[3 * i + 1], derivatives[3 * i + 2]});
            // comp->setDerivative(i, { derivatives[3 * i], derivatives[3 * i + 1], derivatives[3 * i + 2]});
        }
    }

    setBoxDerivativesNoPbc();
}

} // namespace colvar
} // namespace PLMD
