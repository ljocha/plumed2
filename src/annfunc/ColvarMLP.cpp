#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "MultilayerPerceptron.hpp"

#include <string>

namespace PLMD {
namespace colvar {
namespace annfunc {

//+PLUMEDOC COLVAR CMLP
/*
Multilayer perceptron colvar

This component implements a multilayer perceptron collective variable, i.e.,
a neural network with several dense layers. The weights, biases, and activation functions
can be specified as parameters of the component. An optional parameter can be provided
in order to normalize the input data by scalar multiplication.

Unlike the <a href="./_a_n_n.html">ANN function</a>, the inputs of this component are directly atom coordinates,
hence CMLP colvar is considerably faster than ANN function composed with POSITION colvars.

\par Examples

Assume we want to model a multilayer perceptron with three layers of sizes [3, 2, 1]
(note that the size of the first layer is always necesarily a multiple of 3 since there are three input values per atom).
Further, assume there should be the RELU activation function in Layer 1 and the SIGMOID activation function in
Layer 2.

Further assume the weight matrix connecting layers 0 and 1 is
\f{equation*}{
W_{01} = [[1,2,3], [4,5,6]],
\f}
and the weight matrix connecting layers 1 and 2 is
\f{equation*}{
W_{12} = [[7,8]].
\f}

Finaly, let the bias vetors for layers 1 and 2 be the following
\f{align*}{
  b_1 &= [9, 10]\\
  b_2 &= [11].
\f}

Then the following PLUMED code constructs the described multilayer perceptron whose input consists of the three coordinates x,y,z of atom 4
multiplied by \f$0.1\f$.

\plumedfile
CMLP ...
LABEL=ann_cv
ATOMS=4
RESCALE=0.1
SIZES=3,2,1
ACTIVATIONS=RELU,SIGMOID
WEIGHTS0=1,2,3,4,5,6
WEIGHTS1=7,8
BIASES0=9,10
BIASES1=11
... CMLP
\endplumedfile

To access its output component, we use "ann_cv.node-0". If there were more components in the output layer,
we could access them by "ann_cv.node-1", "ann_cv.node-2", etc.

*/
//+ENDPLUMEDOC

class ColvarMLP : public colvar::Colvar
{
private:
  bool useDouble;
  std::vector<AtomNumber> atoms;
  std::unique_ptr<MultilayerPerceptron<double>> dNet;
  std::unique_ptr<MultilayerPerceptron<float>> fNet;

  bool readFlag(const std::string& flagName);
  std::vector<AtomNumber> parseAtoms();
  template <typename Scalar>
  std::unique_ptr<MultilayerPerceptron<Scalar>> parseNetwork();

  template <typename Scalar>
  void calculate(MultilayerPerceptron<Scalar>& net);
public:
  static void registerKeywords(Keywords& keys);
  explicit ColvarMLP(const ActionOptions& ao);
  virtual void calculate();
};

PLUMED_REGISTER_ACTION(ColvarMLP,"CMLP")

void ColvarMLP::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);

  keys.add("atoms","ATOMS","a list of atoms involved in the collectiva variable");
  keys.add("optional", "RESCALE", "a coefficient the input coordinates are multiplied by");
  keys.addFlag("DOUBLE_PREC", false, "whether to use double precision or not");
  keys.add("compulsory", "ACTIVATIONS", "array of activation functions in individual layers, F_1, ..., F_N; options TANH, LINEAR, RELU, SIGMOID");
  keys.add("compulsory", "SIZES", "array of sizes of individual layers, S_0, ..., S_n (S_0 should equal to three times the number of atoms)");
  keys.add("numbered", "WEIGHTS", "array of flattened weight matrices W_1, ..., W_N; matrix W_L of size S_{L-1} x S_L connects layers L-1 and L");
  keys.add("numbered", "BIASES", "array of bias vectors B_1, ..., B_N; vector B_L is added to the inner potential of layer L");

  keys.addOutputComponent("node", "default", "components representing the output nodes of the MLP");
}

std::vector<AtomNumber> ColvarMLP::parseAtoms() {
  std::vector<AtomNumber> atom_buffer;
  parseAtomList("ATOMS", atom_buffer);
  return atom_buffer;
}

bool ColvarMLP::readFlag(const std::string& flagName) {
  bool flag = false;
  parseFlag(flagName, flag);
  return flag;
}

template <typename Scalar>
std::unique_ptr<MultilayerPerceptron<Scalar>> ColvarMLP::parseNetwork() {
  /* sizes of all the layers including the input layer */
  std::vector<int> layerSizes;
  parseVector("SIZES", layerSizes);

  if (layerSizes.size() < 2) error("There should be at least two layers.");
  if (layerSizes.front() != atoms.size() * 3) {
    error("The input layer should contain 3 times as many nodes as there are input atoms.");
  }

  int numLayers = layerSizes.size();

  std::vector<std::string> activationNames;
  std::vector<Scalar (*)(Scalar)> fns{MultilayerPerceptron<Scalar>::linear};
  std::vector<Scalar (*)(Scalar)> d_fns{MultilayerPerceptron<Scalar>::d_linear};

  /* activations in all the layers except for the input layer */
  parseVector("ACTIVATIONS", activationNames);
  for (auto& name : activationNames) {
    if (name == "TANH") {
      fns.push_back(MultilayerPerceptron<Scalar>::tanh);
      d_fns.push_back(MultilayerPerceptron<Scalar>::d_tanh);
    } else if (name == "LINEAR") {
      fns.push_back(MultilayerPerceptron<Scalar>::linear);
      d_fns.push_back(MultilayerPerceptron<Scalar>::d_linear);
    } else if (name == "SIGMOID") {
      fns.push_back(MultilayerPerceptron<Scalar>::sigmoid);
      d_fns.push_back(MultilayerPerceptron<Scalar>::d_sigmoid);
    } else if (name == "RELU") {
      fns.push_back(MultilayerPerceptron<Scalar>::relu);
      d_fns.push_back(MultilayerPerceptron<Scalar>::d_relu);
    } else {
      error("Unknown activation function: " + name + ".");
    }
  }

  if (static_cast<int>(activationNames.size()) != numLayers - 1) {
    error("Wrong number of activation functions given. There should be one for each non-input layer.");
  }

  Scalar rescaleFactor = 1;
  parse("RESCALE", rescaleFactor);

  std::unique_ptr<MultilayerPerceptron<Scalar>> net(new MultilayerPerceptron<Scalar>(layerSizes, fns, d_fns, rescaleFactor));

  std::vector<Scalar> buffer;
  for (int l = 0; l < numLayers-1; ++l) {
    if(!parseNumberedVector("WEIGHTS", l, buffer)) error("Not enough weight matrices provided.");
    if (static_cast<int>(buffer.size()) != layerSizes[l] * layerSizes[l+1]) {
      error("Invalid number of weights between layers " + std::to_string(l) + " and " + std::to_string(l+1) + ".");
    }
    net->setWeights(l, buffer);
  }

  for (int l = 1; l < numLayers; ++l) {
    if(!parseNumberedVector("BIASES", l-1, buffer)) error("Not enough bias vectors provided.");
    if (static_cast<int>(buffer.size()) != layerSizes[l]) error("Invalid number of biases in layer " + std::to_string(l) + ".");
    net->setBiases(l, buffer);
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
    addComponentWithDerivatives("node-" + std::to_string(i));
    componentIsNotPeriodic("node-" + std::to_string(i));
  }

  requestAtoms(atoms);

  checkRead();
}

void ColvarMLP::calculate() {
  if (useDouble) calculate<double>(*dNet);
  else calculate<float>(*fNet);
}

template<typename Scalar>
void ColvarMLP::calculate(MultilayerPerceptron<Scalar>& net) {
  std::vector<Scalar>& input = net.getInput();

  std::vector<Vector> positions = getPositions();
  for (unsigned int i = 0; i < positions.size(); i++) {
    input[3 * i + 0] = positions[i][0];
    input[3 * i + 1] = positions[i][1];
    input[3 * i + 2] = positions[i][2];
  }

  const std::vector<Scalar>& output = net.computeOutput();
  for (uint i = 0; i < output.size(); ++i) {
    std::string compName = "node-" + std::to_string(i);
    Value* comp = getPntrToComponent(compName);

    comp->set(output[i]);
    const std::vector<Scalar>& derivatives = net.computeGradient(i);

    for (unsigned int j = 0; j < positions.size(); j++) {
      setAtomsDerivatives(comp, j, { derivatives[3 * j], derivatives[3 * j + 1], derivatives[3 * j + 2]});
    }

    setBoxDerivativesNoPbc(comp);
  }
}

} // namespace annfunc
} // namespace colvar
} // namespace PLMD
