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
    bool use_double;
    std::vector<AtomNumber> atoms;
    std::unique_ptr<Network<double>> d_net;
    std::unique_ptr<Network<float>> f_net;

    std::vector<AtomNumber> parseAtoms();
    bool readFlag(std::string flag_name);
    template <typename Scalar>
    std::unique_ptr<Network<Scalar>> parseNetwork();

public:
    static void registerKeywords(Keywords& keys);
    ColvarMLP(const ActionOptions& ao);
    void calculate();
    template <typename Scalar>
    void calculate_by_precision(Network<Scalar>& net);
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
}

std::vector<AtomNumber> ColvarMLP::parseAtoms() {
    std::vector<AtomNumber> atoms;
    parseAtomList("ATOMS",atoms);
    return atoms;
}

bool ColvarMLP::readFlag(std::string flag_name) {
    bool flag = false;
    parseFlag("DOUBLE_PREC", flag);
    return flag;
}

template <typename Scalar>
std::unique_ptr<Network<Scalar>> ColvarMLP::parseNetwork() {
    /* sizes of all the layers including the input layer */
    std::vector<int> layer_sizes;
    parseVector("SIZES", layer_sizes);
    
    int num_layers = layer_sizes.size();

    std::vector<std::string> activation_names;
    std::vector<Scalar (*)(Scalar)> fns{Network<Scalar>::linear};
    std::vector<Scalar (*)(Scalar)> d_fns{Network<Scalar>::d_linear};

    /* activations in all the layers except for the input layer */
    parseVector("ACTIVATIONS", activation_names);
    for (auto& name : activation_names) {
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
    

    if (static_cast<int>(activation_names.size()) != num_layers - 1) error("Wrong number of activation functions given.");

    Scalar rescale_factor = 1;
    std::unique_ptr<Network<Scalar>> d_net;
    parse("RESCALE", rescale_factor);

    std::unique_ptr<Network<Scalar>> net(new Network<Scalar>(layer_sizes, fns, d_fns, rescale_factor));

    std::vector<double> buffer;
    for (int l = 0; l < num_layers-1; ++l) {
        if(!parseNumberedVector("WEIGHTS", l, buffer)) error("Not enough weight matrices provided.");
        if (static_cast<int>(buffer.size()) != layer_sizes[l] * layer_sizes[l+1]) error("Invalid number of weights between layers " + to_string(l) + " and " + to_string(l+1) + ".");
        net->set_weights(l, buffer);
    }

    for (int l = 0; l < num_layers-1; ++l) {
        if(!parseNumberedVector("BIASES", l, buffer)) error("Not enough biase vectors provided.");
        if (static_cast<int>(buffer.size()) != layer_sizes[l+1]) error("Invalid number of biases in layer " + to_string(l+1) + ".");
        net->set_biases(l+1, buffer);
    }

    return net;
}

ColvarMLP::ColvarMLP(const ActionOptions& ao)
    : PLUMED_COLVAR_INIT(ao)
    , use_double(readFlag("DOUBLE_PREC"))
    , atoms(parseAtoms())
    , d_net(use_double ? parseNetwork<double>() : nullptr)
    , f_net(use_double ? nullptr : parseNetwork<float>())
{
    requestAtoms(atoms);

    addValueWithDerivatives();
    checkRead();

    getPntrToValue()->setNotPeriodic();
}

void ColvarMLP::calculate() {
    if (use_double) calculate_by_precision<double>(*d_net);
    else calculate_by_precision<float>(*f_net);
}

template<typename Scalar>
void ColvarMLP::calculate_by_precision(Network<Scalar>& net) {
    vector<Scalar>& input = net.get_input();

    vector<Vector> positions = getPositions();
    for (unsigned int i = 0; i < positions.size(); i++) {
        input[3 * i + 0] = positions[i][0];
        input[3 * i + 1] = positions[i][1];
        input[3 * i + 2] = positions[i][2];
    }

    vector<Scalar>& output = net.compute_output();
    setValue(output[0]);

    vector<Scalar>& derivatives = net.compute_gradient(0);

    for (unsigned int i = 0; i < positions.size(); i++) {
        setAtomsDerivatives(i, { derivatives[3 * i], derivatives[3 * i + 1], derivatives[3 * i + 2]});
    }

    /* set box derivatives */
    apply();
}

} // namespace colvar
} // namespace PLMD
