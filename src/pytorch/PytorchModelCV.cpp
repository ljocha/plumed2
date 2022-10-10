/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (c) 2022 of Luigi Bonati.

The pytorch module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The pytorch module is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_LIBTORCH
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

#include "core/PlumedMain.h"
#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"

#include <torch/torch.h>
#include <torch/script.h>

#include <fstream>
#include <cmath>

#include <iostream>

using namespace std;

namespace PLMD {
namespace colvar {
namespace pytorch {

//+PLUMEDOC PYTORCH_FUNCTION PYTORCH_MODEL
/*
Load a PyTorch model compiled with TorchScript.

This can be a function defined in Python or a more complex model, such as a neural network optimized on a set of data. In both cases the derivatives of the outputs with respect to the inputs are computed using the automatic differentiation (autograd) feature of Pytorch.

By default it is assumed that the model is saved as: `model.ptc`, unless otherwise indicated by the `FILE` keyword. The function automatically checks for the number of output dimensions and creates a component for each of them. The outputs are called node-i with i between 0 and N-1 for N outputs.

Note that this function is active only if LibTorch is correctly linked against PLUMED. Please check the instructions in the \ref PYTORCH page.

\par Examples
Load a model called `torch_model.ptc` that takes as input two dihedral angles and returns two outputs.

\plumedfile
#SETTINGS AUXFILE=regtest/pytorch/rt-pytorch_model_2d/torch_model.ptc
phi: TORSION ATOMS=5,7,9,15
psi: TORSION ATOMS=7,9,15,17
model: PYTORCH_MODEL FILE=torch_model.ptc ARG=phi,psi
PRINT FILE=COLVAR ARG=model.node-0,model.node-1
\endplumedfile

*/
//+ENDPLUMEDOC


class PytorchModel :
  public Colvar
{
  unsigned _n_in;
  unsigned _n_out;
  std::vector<AtomNumber> _atoms;
  torch::jit::script::Module _model;

  std::vector<AtomNumber> parseAtoms();

public:
  explicit PytorchModel(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);

  std::vector<float> tensor_to_vector(const torch::Tensor& x);
};

PLUMED_REGISTER_ACTION(PytorchModel,"PYTORCH_MODEL_CV")

std::vector<AtomNumber> PytorchModel::parseAtoms() {
  std::vector<AtomNumber> atom_buffer;
  parseAtomList("ATOMS", atom_buffer);
  return atom_buffer;
}

void PytorchModel::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms","ATOMS","a list of atoms involved in the collectiva variable");
  keys.add("optional","FILE","Filename of the PyTorch compiled model");
  keys.addOutputComponent("node", "default", "Model outputs");
}

// Auxiliary function to transform torch tensors in std vectors
std::vector<float> PytorchModel::tensor_to_vector(const torch::Tensor& x) {
  return std::vector<float>(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
}

PytorchModel::PytorchModel(const ActionOptions&ao)
  : PLUMED_COLVAR_INIT(ao)
  , _atoms(parseAtoms())
{ //print pytorch version

  // request atoms
  requestAtoms(_atoms);

  //number of inputs of the model
  _n_in=_atoms.size() * 3;

  //parse model name
  std::string fname="model.ptc";
  parse("FILE",fname);

  //deserialize the model from file
  try {
    _model = torch::jit::load(fname);
  }
  //if an error is thrown check if the file exists or not
  catch (const c10::Error& e) {
    std::ifstream infile(fname);
    bool exist = infile.good();
    infile.close();
    if (exist) {
      // print libtorch version
      std::stringstream ss;
      ss << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "." << TORCH_VERSION_PATCH;
      std::string version;
      ss >> version; // extract into the string.
      plumed_merror("Cannot load FILE: '"+fname+"'. Please check that it is a Pytorch compiled model (exported with 'torch.jit.trace' or 'torch.jit.script') and that the Pytorch version matches the LibTorch one ("+version+").");
    }
    else {
      plumed_merror("The FILE: '"+fname+"' does not exist.");
    }
  }

  checkRead();

  //check the dimension of the output
  log.printf("Checking output dimension:\n");
  std::vector<float> input_test (_n_in);
  torch::Tensor single_input = torch::tensor(input_test);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( single_input );
  torch::Tensor output = _model.forward( inputs ).toTensor();
  std::vector<float> cvs = this->tensor_to_vector (output);
  _n_out=cvs.size();

  //create components
  for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    addComponentWithDerivatives( name_comp );
    componentIsNotPeriodic( name_comp );
  }

  //print log
  //log.printf("Pytorch Model Loaded: %s \n",fname);
  log.printf("Number of input: %d \n",_n_in);
  log.printf("Number of outputs: %d \n",_n_out);
  log.printf("  Bibliography: ");
  log<<plumed.cite("Bonati, Rizzi and Parrinello, J. Phys. Chem. Lett. 11, 2998-3004 (2020)");
  log.printf("\n");

}

void PytorchModel::calculate() {

  //retrieve arguments
  std::vector<Vector> positions = getPositions();
  std::vector<float> current_S(_n_in);
  for(unsigned i=0; i<positions.size(); i++) {
    current_S[3 * i + 0] = positions[i][0];
    current_S[3 * i + 1] = positions[i][1];
    current_S[3 * i + 2] = positions[i][2];
  }
    
  //convert to tensor
  torch::Tensor input_S = torch::tensor(current_S);
  input_S.set_requires_grad(true);
  //convert to Ivalue
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back( input_S );
  //calculate output
  torch::Tensor output = _model.forward( inputs ).toTensor();
  //set CV values
  std::vector<float> cvs = this->tensor_to_vector (output);
  for(unsigned j=0; j<_n_out; j++) {
    string name_comp = "node-"+std::to_string(j);
    getPntrToComponent(name_comp)->set(cvs[j]);
  }

  std::cout << "num out=" << _n_out << std::endl;
  //derivatives
  for(unsigned j=0; j<_n_out; j++) {
    // expand dim to have shape (1,_n_out)
    auto grad_outputs = torch::ones({1});
    // calculate derivatives with automatic differentiation
    std::cout << "compute graphs" << std::endl;
    auto gradient = torch::autograd::grad({output.slice(/*dim=*/0, /*start=*/j, /*end=*/j+1)},
    {input_S},
    /*grad_outputs=*/ {grad_outputs},
    /*retain_graph=*/true,
    /*create_graph=*/false);
    // add dimension
    auto grad = gradient[0];
    //convert to vector
    std::vector<float> der = this->tensor_to_vector ( grad );

    string name_comp = "node-"+std::to_string(j);
    Value* comp = getPntrToComponent(name_comp);

    //set derivatives of component j
    for (unsigned int i = 0; i < positions.size(); i++) {
      setAtomsDerivatives(comp, i, { der[3 * i], der[3 * i + 1], der[3 * i + 2] });
    }

    setBoxDerivativesNoPbc(comp);
  }
}
}
}
}

#endif
