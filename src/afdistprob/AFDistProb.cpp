#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "tools/Matrix.h"
#include "tools/Vector.h"

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR AF_DISTPROB
/*
\plumedfile
AF_DISTPROB ...
LABEL=prp
ATOMS=1,5
LAMBDA=1000
EPSILON=0.000001
DISTANCES=0.110,0.111,0.112,0.113
PROB_MATRIX0=0,0.1,0.1,0
PROB_MATRIX1=0,0.4,0.4,0
PROB_MATRIX2=0,0.3,0.3,0
PROB_MATRIX3=0,0.2,0.2,0
... AF_DISTPROB
\endplumedfile
*/
//+ENDPLUMEDOC


template <typename T>
class _3_Tensor {
  using storage_type = std::vector<std::vector<std::vector<T>>>;

  storage_type data;
  public:
    _3_Tensor() = default;

    _3_Tensor(size_t w, size_t h, size_t d)
    : data(storage_type(w, std::vector<std::vector<T>>(h, std::vector<T>(d)))) {}

    std::vector<std::vector<T>>& operator[](int i) { return data[i]; }
    const std::vector<std::vector<T>>& operator[](int i) const { return data[i]; }
};


class AFDistProb : public Colvar {
  using Matrix = PLMD::Matrix<double>;
  using Tensor = _3_Tensor<double>;

  std::vector<AtomNumber> atoms;
  Tensor probs;
  std::vector<double> dists;

  double lambda;
  double epsilon;

private:
  std::pair<double, double> interpolate(const std::vector<double>& probs, double dist);

public:
  explicit AFDistProb(const ActionOptions&);
// active methods:
  void calculate() override;
/// Register all the keywords for this action
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(AFDistProb,"AF_DISTPROB")

AFDistProb::AFDistProb(const ActionOptions&ao)
  : PLUMED_COLVAR_INIT(ao)
  , lambda(1)
  , epsilon(0)
{
  addValueWithDerivatives(); setNotPeriodic();
  
  parseAtomList("ATOMS", atoms);
  parseVector("DISTANCES", dists);
  parse("LAMBDA", lambda);
  parse("EPSILON", epsilon);

  probs = Tensor(atoms.size(), atoms.size(), dists.size());
  std::vector<double> prob_vector(atoms.size() * atoms.size());
  
  for (size_t d = 0; d < dists.size(); ++d) {
    parseNumberedVector("PROB_MATRIX", d, prob_vector);
    
    /** Fill logit matrix for particular distance */
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = 0; j < atoms.size(); ++j) {
        probs[i][j][d] = prob_vector[i * atoms.size() + j];
      }
    }

    /** Check whether the matrix is symmetric */
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = 0; j < atoms.size(); ++j) {
        if (probs[i][j][d] != probs[j][i][d]) error("probs matrix is not symmetric");
      }
    }
  }

  requestAtoms(atoms);

  checkRead();
}

void AFDistProb::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );

  // componentsAreNotOptional(keys);

  keys.add("atoms","ATOMS","a list of atoms whose coordinates are used to compute the probability.");
  keys.add("compulsory", "DISTANCES", "a list of distances.");
  keys.add("numbered", "PROB_MATRIX", "a flattened matrix of probabilities for given distance from the DISTANCES list.");
  // keys.reset_style("PROB_MATRIX", "compulsory");
  keys.add("optional", "LAMBDA", "a smoothness parameter of the property map.");
  keys.add("optional", "EPSILON", "a small positive constant ensuring numerical stability of division.");
}

/**
 * Currently using Property map interpolation
 */
std::pair<double, double> AFDistProb::interpolate(const std::vector<double>& probs, double dist) {
  double sum = epsilon;
  double weighted_sum = 0;

  double d_sum = 0;
  double d_weighted_sum = 0;

  for (size_t d = 0; d < probs.size(); ++d) {
    double t = exp(-lambda * pow(dists[d] - dist, 2));
    
    sum += t;
    weighted_sum += t * probs[d];

    d_sum += t * (dists[d] - dist);
    d_weighted_sum += t * (dists[d] - dist) * probs[d];
  }

  d_sum *= 2 * lambda;
  d_weighted_sum *= 2 * lambda;

  return { weighted_sum / sum, (d_weighted_sum * sum - weighted_sum * d_sum) / (sum * sum) };
}

// calculator
void AFDistProb::calculate() {
  std::vector<PLMD::Vector> positions = getPositions();

  Matrix real_dists(atoms.size(), atoms.size());
  for (size_t i = 0; i < atoms.size(); ++i) {
    for (size_t j = 0; j < atoms.size(); ++j) {
      real_dists(i, j) = real_dists(j, i) = delta(positions[i], positions[j]).modulo();
    }
  }

  Matrix gradient(atoms.size(), atoms.size());

  double prob_sum = 0;
  for(size_t i = 0; i < atoms.size(); ++i) {
    for(size_t j = i + 1; j < atoms.size(); ++j) {
      auto r = interpolate(probs[i][j], real_dists(i, j));
      prob_sum += r.first;
      gradient(i, j) = gradient(j, i) = r.second;
    }
  }

  for ( size_t i = 0; i < atoms.size(); ++i) {
    Vector derivatives;
    for ( size_t j = 0; j < atoms.size(); ++j) {
      if (i == j) continue;
      derivatives += delta(positions[j], positions[i]) * (gradient(i, j) / real_dists(i, j));
    }

    setAtomsDerivatives(i, derivatives);
  }
  setBoxDerivativesNoPbc();
  setValue(prob_sum);
}

} // end of namespace colvar
} // end of namespace PLMD



