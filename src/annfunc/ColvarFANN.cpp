#include "../colvar/Colvar.h"
#include "../colvar/ActionRegister.h"

#include <cassert>
#include <string>
#include <cmath>
#include <iostream>

#include "./network.hpp"

using namespace std;

// #define DEBUG
// #define DEBUG_2
// #define DEBUG_3

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR ColvarFANN
/*
Fast Artificial Neural Network colvar

Computes the function of two layer perceptron with positions of atoms as input and one output.
Optionally can rescale the input by a specified constant.

\plumedfile
FANN ...
LABEL=ann_cv1
ATOMS=1,4,12
RESCALE=0.100041
WEIGHTS_INPUT_HIDDEN=1.35345,2.353535,-3.242425,4.23424,5.25425,6.252343
WEIGHTS_HIDDEN_OUTPUT=0.125232,0.224323
BIASES_HIDDEN=-0.005190,0.001307
BIASES_OUTPUT=0.213432
... FANN
\endplumedfile

*/
//+ENDPLUMEDOC


class ColvarFANN : public Colvar
{
private:
    std::vector<AtomNumber> atoms;
    Network net;

    std::vector<AtomNumber> parseAtoms();
    Network parseNetwork();

public:
    static void registerKeywords(Keywords& keys);
    ColvarFANN(const ActionOptions& ao);
    virtual void calculate();
};

PLUMED_REGISTER_ACTION(ColvarFANN,"FANN")

void ColvarFANN::registerKeywords( Keywords& keys ) {
    Colvar::registerKeywords(keys);

    keys.add("atoms","ATOMS","the list of atoms involved in this collective variable");
    keys.add("optional", "RESCALE", "a coefficient by which the input is multiplied");

    keys.add("compulsory", "WEIGHTS_INPUT_HIDDEN", "array of flattened weights connecting the input and the hidden layer");
    keys.add("compulsory", "WEIGHTS_HIDDEN_OUTPUT", "array of flattened weights connecting the hidden and the output layer");

    keys.add("compulsory", "BIASES_HIDDEN", "array of biases used in the hidden layer");
    keys.add("compulsory", "BIASES_OUTPUT", "array of biases used in the output layer");

    // since v2.2 plumed requires all components be registered
    // keys.addOutputComponent("node", "default", "components of ColvarFANN outputs");
}

std::vector<AtomNumber> ColvarFANN::parseAtoms() {
    std::vector<AtomNumber> atoms;
    parseAtomList("ATOMS",atoms);
    return atoms;
}

Network ColvarFANN::parseNetwork() {
    vector<double> weights_input_hidden;
    vector<double> weights_hidden_output;
    vector<double> biases_hidden;
    double biases_output;
    double rescale_factor;

    parse("RESCALE", rescale_factor);
    parseVector("WEIGHTS_INPUT_HIDDEN", weights_input_hidden);
    parseVector("WEIGHTS_HIDDEN_OUTPUT", weights_hidden_output);
    parseVector("BIASES_HIDDEN", biases_hidden);
    parse("BIASES_OUTPUT", biases_output);

    int input_size = static_cast<int>(atoms.size() * 3);
    int hidden_size = biases_hidden.size();

    if (static_cast<int>(weights_input_hidden.size()) != input_size * hidden_size) error("Invalid number of input-hidden weights.");
    if (static_cast<int>(weights_hidden_output.size()) != hidden_size) error("Invalid number of hiden-output weights.");
    if (static_cast<int>(biases_hidden.size()) != hidden_size) error("Invalid number of biases in the hidden layer");

    return Network(input_size, hidden_size, rescale_factor, weights_input_hidden, weights_hidden_output, biases_hidden, biases_output);
}

ColvarFANN::ColvarFANN(const ActionOptions& ao)
    : PLUMED_COLVAR_INIT(ao)
    , atoms(parseAtoms())
    , net(parseNetwork())
{
    requestAtoms(atoms);

    addValueWithDerivatives();
    checkRead();

    getPntrToValue()->setNotPeriodic();
}

void ColvarFANN::calculate() {
    vector<double>& input = net.get_input();
    vector<double>& derivatives = net.get_input_gradient();

    vector<Vector> positions = getPositions();
    for (unsigned int i = 0; i < positions.size(); i++) {
        input[3 * i + 0] = positions[i][0];
        input[3 * i + 1] = positions[i][1];
        input[3 * i + 2] = positions[i][2];
    }

    setValue(net.forth_and_back_propagation());

    for (unsigned int i = 0; i < positions.size(); i++) {
        setAtomsDerivatives(i, { derivatives[3 * i], derivatives[3 * i + 1], derivatives[3 * i + 2]});
    }

    /* set box derivatives */
    apply();
}

} // namespace colvar
} // namespace PLMD
