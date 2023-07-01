#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Matrix.h"
#include "tools/Vector.h"

namespace PLMD {
namespace colvar {
namespace afed {

//+PLUMEDOC COLVAR AFED
/*
AFDE \cite Spiwok2022 is an abbreviation of AlphaFold expected difference. It is a collective variable whose value
captures the likeliness that the internal coordinates of a given protein take their current values
with respect to the distribution predicted by AlphaFold.

More precisely, for every pair of residual atoms \f$A_i, A_j\f$, AlphaFold is able to generate a probabilistic distribution \f$P_{i,j}\f$
over possible distances between them.
The space of possible distances between two atoms is discretized into \f$m\f$ bins centred at points \f$d_1, \ldots, d_m\f$,
which allows us to write the distribution \f$P_{i,j}\f$ as a vector of probabilities
\f$P_{i,j}(d_1), \ldots, P_{i,j}(d_m)\f$ over the individual bins.
Based on this data, AFED computes the expected similarity between the current spacial configuration of the molecule
and its random spacial arrangement sampled according to \f$P\f$.
Namely, the colvar computes the expected number of pairs \f$(i,j)\f$ such that the bin \f$d_{i,j}\f$ corresponding to the actual
distance between atoms \f$A_i\f$ and \f$A_j\f$ coincide with a random bin \f$d'_{i,j}\f$ drawn from the distribution \f$P_{i,j}\f$.

Thus the value of the collective variable is almost equal to the sum \f$\sum_{1 \leq i < j \leq n}P_{i,j}(d_{i,j})\f$.
However, since a collective variable needs to be differentiable,
the colvar in fact computes a weighted average of probabilities of all the bins. This can be viewed as a kind of interpolation
that shifts the positions of the bins so that the actual distance of the pair of residues always lies in the middle of its bin.
The interpolation is computed by the property map as described in
the <a href="./_p_r_o_p_e_r_t_y_m_a_p.html">propery map collective variable</a>.

Altogether, the output of the collective variable is the following sum
\f{equation*}{
  \sum_{0 \leq i < j < n} \tilde{P}_{i,j}(d(A_i, A_j)),
\f}
where \f$d(A_i, A_j)\f$ means the distance between atoms \f$A_i\f$ and \f$A_j\f$,
and \f$ \tilde{P}_{i,j}(d(A_i, A_j)) \f$ denotes the weighted sum
\f{equation*}{
\tilde{P}_{i,j}(d) = \frac{\sum_{k=1}^{m} P_{i,j}(d_k)e^{-\lambda(d - d_k)}}{\varepsilon + \sum_{k=1}^{m} e^{-\lambda(d - d_k)}}.
\f}

\par Examples

The intended way of generating the tensor of probabilities \f$P\f$ is via the algorithm AlphaFold 2 \cite Jumper2021.

Here is an example of a usage of AFED colvar.
\plumedfile
AFED ...
ATOMS=1,3,4
LAMBDA=1000
EPSILON=0.000001
DISTANCES=0.110,0.111,0.112
PROB_MATRIX0=0,0.1,0.2,0.1,0,0.3,0.2,0.3,0
PROB_MATRIX1=0,0.75,0.55,0.75,0,0.35,0.55,0.35,0
PROB_MATRIX2=0,0.15,0.25,0.15,0,0.35,0.25,0.35,0
... AFED
\endplumedfile
There are three distance bins in the example located around points \f$0.110\f$, \f$0.111\f$, and \f$0.112\f$.
Therefore, there are three matrices `PROB_MATRIX0`, `PROB_MATRIX1`, and `PROB_MATRIX2`. Since we are working with three atoms,
the matrices have order three. For example, the list `0,0.1,0.2,0.1,0,0.3,0.2,0.3,0` represents the matrix
\f{pmatrix}{
0 & 0.1 & 0.2\\
0.1 & 0 & 0.3\\
0.2 & 0.3 & 0
\f}
The values `PROB_MATRIX0[i][j]`, `PROB_MATRIX1[i][j]`, `PROB_MATRIX2[i][j]` are the probabilities
that the distances between atoms \f$A_i\f$ and \f$A_j\f$ lie in the bins centered at points
\f$0.110\f$,\f$0.111\f$, and \f$0.112\f$ respectively. Note that these numbers should sum up to one
except for the diagonal entries, which should be always zero.


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


class AFED : public Colvar {
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
  explicit AFED(const ActionOptions&);
// active methods:
  void calculate() override;
  bool isPeriodic() { return false; }
/// Register all the keywords for this action
  static void registerKeywords( Keywords& keys );
};

PLUMED_REGISTER_ACTION(AFED,"AFED")

AFED::AFED(const ActionOptions&ao)
  : PLUMED_COLVAR_INIT(ao)
  , lambda(1)
  , epsilon(1e-8)
{
  addValueWithDerivatives(); setNotPeriodic();

  parseAtomList("ATOMS", atoms);
  parseVector("DISTANCES", dists);
  for (double d : dists) {
    if (d <= 0) error("All distances should be positive.");
  }


  parse("LAMBDA", lambda);
  parse("EPSILON", epsilon);
  if (epsilon < 0) error("EPSILON should be non-negative.");
  if (epsilon < 0) error("LAMBDA should be non-negative.");

  probs = Tensor(atoms.size(), atoms.size(), dists.size());
  std::vector<double> prob_vector(atoms.size() * atoms.size());

  for (size_t d = 0; d < dists.size(); ++d) {
    parseNumberedVector("PROB_MATRIX", d, prob_vector);
    if (prob_vector.size() != atoms.size() * atoms.size()) {
      error("The matrix PROB_MATRIX" + std::to_string(d) + " has invalid size.");
    }

    /** Fill the probability matrix of a particular bin. */
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = 0; j < atoms.size(); ++j) {
        probs[i][j][d] = prob_vector[i * atoms.size() + j];
      }
    }

    /** Check whether the matrix is symmetric */
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = 0; j < atoms.size(); ++j) {
        if (probs[i][j][d] != probs[j][i][d]) error("The matrix PROB_MATRIX" + std::to_string(d) + " is not symmetric.");
      }
    }
  }

  requestAtoms(atoms);

  checkRead();
}

void AFED::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );

  keys.add("atoms","ATOMS","A list of residua of the molecule");
  keys.add("compulsory", "DISTANCES", "A list of centres of the distance bins");
  keys.add("numbered", "PROB_MATRIX", "A flattened matrix of probabilities for every bin described in DISTANCES. `PROB_MATRIXk` corresponds to the k-th bin");
  keys.add("optional", "LAMBDA", "A smoothness parameter of the property map. The default value is 1");
  keys.add("optional", "EPSILON", "A small positive constant ensuring the numerical stability of division. The default value is 1e-8.");
}

/**
 * Interpolate using the property map
 */
std::pair<double, double> AFED::interpolate(const std::vector<double>& probs, double dist) {
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


void AFED::calculate() {
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

} // end of namespace afed
} // end of namespace colvar
} // end of namespace PLMD
