#include "colvar/Colvar.h"
#include "colvar/ActionRegister.h"
#include "tools/Matrix.h"
#include "tools/Vector.h"

namespace PLMD {
namespace colvar {
namespace afdistprob {

//+PLUMEDOC COLVAR AF_DISTPROB
/*
The collective variable is used to calculate the probability of a molecule having a particular shape
given a probability distribution of distances between every pair of atoms of the molecule.
This module was developed to be used together with the AlphaFold neural network
that is capable of generating such matrices for protain molecules \cite Jumper2021.

The input for the collective variable consists of a vector of atoms ATOMS, list of distances DISTANCES, the matrix of probabilities PROB_MATRIXn
for every listed distance DISTANCES[i], two numerical parameters LAMBDA and EPSILON.
ATOMS is a vector of atoms whose shape probability is being computed.
DISTANCES is a vector that enumerates a set of possible distances of two atoms for which the probabilities are precomputed.
PROB_MATRIXn is a flattened (symetric) matrix of probabilities where the entry PROB_MATRIXn[i][j] denotes the probability,
that the two atoms ATOMS[i] and ATOMS[j] have distance DISTANCES[n].

The interpolation of this distribution is computed using the property map as described
in the <a href="https://www.plumed.org/doc-v2.6/user-doc/html/_p_r_o_p_e_r_t_y_m_a_p.html">propery map collective variable</a>.
LAMBDA is the parameter as in the definition of the property map.
EPSILON is a positive number close to zero ensuring numerical stability.

The output of the collective variable is the following sum:
\f$$\sum_{i,j} I(i,j,d(ATOM[i], ATOM[j]))\f$$
where \f$ d(ATOM[i], ATOM[j]) \f$ is the distance between atoms \f$ i \f$ and \f$ j \f$ and
\f$ I(i,j,l) \f$ is the interpolation of the values \f$ DISTANCES0[i][j],\ldots, DISTANCESn[i][j]\f$ using the property map.

\par Examples
Since the actual product of probabilities of individual pairwise distances is for various reasons computationally inpractical,
the collective variable produces a sum of the probabilities insted. See this paper for more details.

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
  bool isPeriodic(){ return false; }
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
    
    /** Fill probability matrix for particular distance */
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

  keys.add("atoms","ATOMS","a list of atoms whose coordinates are used to compute the probability.");
  keys.add("compulsory", "DISTANCES", "a list of distances.");
  keys.add("numbered", "PROB_MATRIX", "a flattened matrix of probabilities for given distance from the DISTANCES list.");
  keys.add("optional", "LAMBDA", "a smoothness parameter of the property map.");
  keys.add("optional", "EPSILON", "a small positive constant ensuring numerical stability of division.");
}

/**
 * Interpolate using the property map
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

} // end of namespace afdistprob
} // end of namespace colvar
} // end of namespace PLMD
