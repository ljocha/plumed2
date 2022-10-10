#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "LibTorchModel.hpp"

#include <string>

namespace PLMD {
namespace colvar {
namespace annfunc {

//+PLUMEDOC COLVAR PT
/*
Pytorch colvar


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

class ColvarPTM : public colvar::Colvar
{
private:
  std::vector<AtomNumber> atoms;
  std::unique_ptr<LibTorchModel> model;

  bool readFlag(const std::string& flagName);
  std::vector<AtomNumber> parseAtoms();
  std::unique_ptr<LibTorchModel> parseModel(int inputSize);

public:
  static void registerKeywords(Keywords& keys);
  explicit ColvarPTM(const ActionOptions& ao);
  virtual void calculate();
};

PLUMED_REGISTER_ACTION(ColvarPTM,"TorchModel")

void ColvarPTM::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords(keys);

  keys.add("atoms","ATOMS","a list of atoms involved in the collectiva variable");
  keys.add("compulsory", "MODEL_FILE_NAME", "name of the file containing the model");
  keys.addOutputComponent("node", "default", "components representing the output nodes of the PyTorch model");
}

std::vector<AtomNumber> ColvarPTM::parseAtoms() {
  std::vector<AtomNumber> atom_buffer;
  parseAtomList("ATOMS", atom_buffer);
  return atom_buffer;
}

bool ColvarPTM::readFlag(const std::string& flagName) {
  bool flag = false;
  parseFlag(flagName, flag);
  return flag;
}

std::unique_ptr<LibTorchModel> ColvarPTM::parseModel(int inputSize) {
  std::string modelFileName;
  parse("MODEL_FILE_NAME", modelFileName);

  std::unique_ptr<LibTorchModel> model = std::make_unique<LibTorchModel>(modelFileName, inputSize);

  return model;
}

ColvarPTM::ColvarPTM(const ActionOptions& ao)
  : PLUMED_COLVAR_INIT(ao)
  , atoms(parseAtoms())
  , model(parseModel(atoms.size() * 3))
{
  int outputSize = model->outputSize();

  for (int i = 0; i < outputSize; ++i) {
    addComponentWithDerivatives("node-" + std::to_string(i));
    componentIsNotPeriodic("node-" + std::to_string(i));
  }

  requestAtoms(atoms);

  checkRead();
}

void ColvarPTM::calculate() {
  auto& input = model->getInput();

  std::vector<Vector> positions = getPositions();
  for (unsigned int i = 0; i < positions.size(); i++) {
    input[3 * i + 0] = positions[i][0];
    input[3 * i + 1] = positions[i][1];
    input[3 * i + 2] = positions[i][2];
  }

  auto output = model->computeOutput();
  for (uint i = 0; i < output.sizes()[0]; ++i) {
    std::string compName = "node-" + std::to_string(i);
    Value* comp = getPntrToComponent(compName);

    comp->set(output[i].item<float>());
    auto derivatives = model->computeGradient(i);

    for (unsigned int j = 0; j < positions.size(); j++) {
      setAtomsDerivatives(comp, j, { derivatives[3 * j].item<float>(), derivatives[3 * j + 1].item<float>(), derivatives[3 * j + 2].item<float>()});
    }

    setBoxDerivativesNoPbc(comp);
  }
}

} // namespace annfunc
} // namespace colvar
} // namespace PLMD
