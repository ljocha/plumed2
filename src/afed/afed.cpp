#include "colvar/Colvar.h"
#include "core/ActionRegister.h"
#include "tools/Matrix.h"
#include "tools/Vector.h"

#include <fstream>

namespace PLMD {
namespace colvar {
namespace afed {

//+PLUMEDOC COLVAR AFED
/*
AFED [Spiwok2022] is an abbreviation of AlphaFold expected difference. It is a collective variable whose value
captures the likeliness that distances of alpha carbon pairs of the protein take their current values
with respect to the distribution predicted by AlphaFold.

More precisely, for every CA pair \f$A_i, A_j\f$ of residua \f$i,j\f$, AlphaFold generates a probabilistic distribution \f$P_{i,j}\f$
over possible distances between them; technically, it is discretized 
into \f$m\f$ bins centred at points \f$d_1, \ldots, d_m\f$,
storing the distribution \f$P_{i,j}\f$ for each \f$i,j\f$ as a vector of probabilities
\f$P_{i,j}(d_1), \ldots, P_{i,j}(d_m)\f$ over the individual bins.
Based on this data, AFED computes a probabilistic measure how AlphaFold would favor the current protein configuration
(see \cite Spiwok2022 for more details and discussion). 

Since a collective variable has to be differentiable, softmax-like smoothing similar to
the <a href="./_p_r_o_p_e_r_t_y_m_a_p.html">PROPERTYMAP</a> collective variable.
is applied on the discretized intermediate results,
yielding the overall formula
\f{equation*}{
  \sum_{0 \leq i < j < n} \tilde{P}_{i,j}(d(A_i, A_j)),
\f}
where \f$d(A_i, A_j)\f$ means the distance between atoms \f$A_i\f$ and \f$A_j\f$,
and \f$ \tilde{P}_{i,j}(d(A_i, A_j)) \f$ denotes the weighted sum
\f{equation*}{
\tilde{P}_{i,j}(d) = \frac{\sum_{k=1}^{m} P_{i,j}(d_k)e^{-\lambda(d - d_k)}}{\varepsilon + \sum_{k=1}^{m} e^{-\lambda(d - d_k)}}.
\f}

\f$\varepsilon\f$ is a small constant preventing numerical instability in the case all the exponential terms approach zero.

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

By default, all-to-all ATOMS distances are considered in the equation above. 
This can be restricted further to enumerated atom pairs only with ATOMS1, ATOMS2 keywords, or groupwise pairs with GROUP, GROUPA, GROUPB, similarly to 
<a href="./_d_i_s_t_a_n_c_e_s.html">DISTANCES</a> collective variable.

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

  std::vector<AtomNumber> all_atoms;
  std::vector<std::vector<bool>> dist_pairs;
  
  Tensor probs;
  std::vector<double> dists;

  double lambda;
  double epsilon;

private:
  std::pair<double, double> interpolate(const std::vector<double>& probs, double dist);
  std::vector<size_t> find_indices(const std::vector<AtomNumber> &) const;

  std::fstream bonz;
  long step = 0;

public:
  explicit AFED(const ActionOptions&);
// active methods:
  void calculate() override;
  bool isPeriodic() { return false; }
/// Register all the keywords for this action
  static void registerKeywords( Keywords& keys );
  void makeWhole();
};

PLUMED_REGISTER_ACTION(AFED,"AFED")

AFED::AFED(const ActionOptions&ao)
  : PLUMED_COLVAR_INIT(ao)
  , lambda(1)
  , epsilon(1e-8)
{
  addValueWithDerivatives(); setNotPeriodic();

  parseAtomList("ATOMS", all_atoms);

  auto size = all_atoms.size();
  dist_pairs.resize(size);
  for (auto &r: dist_pairs) r.assign(size,false);

  parseVector("DISTANCES", dists);
  for (double d : dists) {
    if (d <= 0) error("All distances should be positive.");
  }


  parse("LAMBDA", lambda);
  parse("EPSILON", epsilon);
  if (epsilon < 0) error("EPSILON should be non-negative.");
  if (epsilon < 0) error("LAMBDA should be non-negative.");

  probs = Tensor(all_atoms.size(), all_atoms.size(), dists.size());
  std::vector<double> prob_vector(all_atoms.size() * all_atoms.size());

  for (size_t d = 0; d < dists.size(); ++d) {
    parseNumberedVector("PROB_MATRIX", d, prob_vector);
    if (prob_vector.size() != all_atoms.size() * all_atoms.size()) {
      error("The matrix PROB_MATRIX" + std::to_string(d) + " has invalid size.");
    }

    /** Fill the probability matrix of a particular bin. */
    for (size_t i = 0; i < all_atoms.size(); ++i) {
      for (size_t j = 0; j < all_atoms.size(); ++j) {
        probs[i][j][d] = prob_vector[i * all_atoms.size() + j];
      }
    }

    /** Check whether the matrix is symmetric */
    for (size_t i = 0; i < all_atoms.size(); ++i) {
      for (size_t j = 0; j < all_atoms.size(); ++j) {
        if (probs[i][j][d] != probs[j][i][d]) error("The matrix PROB_MATRIX" + std::to_string(d) + " is not symmetric.");
      }
    }
  }

  std::vector<AtomNumber> atoms1, atoms2, group, groupa, groupb;
  parseAtomList("ATOMS1",atoms1);
  parseAtomList("ATOMS2",atoms2);
  parseAtomList("GROUP",group);
  parseAtomList("GROUPA",groupa);
  parseAtomList("GROUPB",groupb);

  auto idx = find_indices(all_atoms);

  if (atoms1.size() != atoms2.size()) error("ATOMS1 and ATOMS2 must be the same length");
  if (atoms1.size() > 0) {
    log.printf("AFED: using ATOMS1/2\n");
    if (group.size() > 0 || groupa.size() > 0 || groupb.size() > 0) error("ATOMS1/2 is mutually exclusive with GROUP/A/B");

    for (size_t i=0; i<atoms1.size(); i++) {
      if (atoms1[i] == atoms2[i]) error("Single atom can't make a pair");
      dist_pairs[idx[atoms1[i].index()]][idx[atoms2[i].index()]] = dist_pairs[idx[atoms2[i].index()]][idx[atoms1[i].index()]] = true;
    }
  }
  else if (group.size() > 0) {
    log.printf("AFED: using GROUP\n");
    if (groupa.size() > 0 || groupb.size() > 0) error("GROUP is mutually exclusive with GROPUA/B");
    if (group.size() == 1) error("GROUP must contain at least two atoms");

    for (size_t i=0; i<group.size(); i++) 
      for (size_t j=0; j<group.size(); j++) 
        dist_pairs[idx[group[i].index()]][idx[group[j].index()]] = (i != j);
  }
  else if (groupa.size() > 0) {
    if (groupb.size() == 0) error("GROUPB must not be empty");
    log.printf("AFED: using GROUPA/B\n");

    for (size_t i=0; i<groupa.size(); i++)
      for (size_t j=0; j<groupb.size(); j++)
        if (groupa[i] != groupb[j]) dist_pairs[idx[groupa[i].index()]][idx[groupb[j].index()]] = dist_pairs[idx[groupb[j].index()]][idx[groupa[i].index()]] = true;

  }
  else {
    log.printf("AFED: using default all-to-all ATOMS\n");
    for (size_t i=0; i<size; i++) 
      for (size_t j=0; j<size; j++)
        dist_pairs[i][j] = (i != j);
  }

  requestAtoms(all_atoms);

  checkRead();
  bonz.open("bonz.xyz",std::fstream::out);
}

void AFED::registerKeywords( Keywords& keys ) {
  Colvar::registerKeywords( keys );

  keys.add("atoms","ATOMS","List of atoms (CA typically) PROB_MATRIX refers to");
  keys.add("compulsory", "DISTANCES", "A list of centres of the distance bins");
  keys.add("numbered", "PROB_MATRIX", "A flattened matrix of probabilities for every bin described in DISTANCES. `PROB_MATRIXk` corresponds to the k-th bin");
  keys.add("optional", "LAMBDA", "A smoothness parameter of the property map. The default value is 1");
  keys.add("optional", "EPSILON", "A small positive constant ensuring the numerical stability of division. The default value is 1e-8.");
  keys.add("atoms", "ATOMS1", "Explicitely enumerated atom pairs, left atoms");
  keys.add("atoms", "ATOMS2", "Explicitely enumerated atom pairs, right atoms");
  keys.add("atoms", "GROUP", "Atoms to construct any-any pairs");
  keys.add("atoms", "GROUPA", "Atoms to construct pairs, left side");
  keys.add("atoms", "GROUPB", "Atoms to construct pairs, right side");
}

std::vector<size_t> AFED::find_indices(const std::vector<AtomNumber> &atoms) const
{
  std::vector<size_t> r;
  size_t i = 0;
  for (auto &a: atoms) {
    if (r.size() < a.index()) r.resize(a.index()+1);
    r[a.index()] = i++;
  }
    
  return r;
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

void AFED::makeWhole() {
  auto positions = getPositions();
  long crit = 599;
  if (step == crit) log.printf("pbcdump: ");
  for(unsigned j=0; j<positions.size()-1; ++j) {
    const Vector & first (positions[j]);
    Vector & second (positions[j+1]);
    auto od = delta(first,second);
    auto d = pbcDistance(first,second);
    second=first+d;
    if (step == crit) log.printf("%f %f %f ",od[0],od[1],od[2]);
  }
  if (step == crit) log.printf("\n");
}

void AFED::calculate() {
//  makeWhole();
  std::vector<PLMD::Vector> positions = getPositions();

/*
// debug
  for (auto p: positions) {
    bonz << "X " << p[0] << " " << p[1] << " " << p[2] << std::endl;
  }

//   setBoxDerivativesNoPbc();
  setValue(0);
  return;
// end debug
*/

  auto size = all_atoms.size();
  bool first = true;

  Matrix real_dists(size, size);
  for (size_t i = 0; i < size; ++i)
    for (size_t j = 0; j < i; ++j)
      if (dist_pairs[i][j]) {
        real_dists(i, j) = real_dists(j, i) = delta(positions[i], positions[j]).modulo();
        if (first) {
//          log.printf("good dist(%lu, %lu) = %f\n",i,j,real_dists(i,j));
          first = false;
        }
      }

  Matrix gradient(size, size);

  double prob_sum = 0;
  for(size_t i = 0; i < size; ++i)
    for(size_t j = 0; j < i; ++j)
      if (dist_pairs[i][j]) {
        auto r = interpolate(probs[i][j], real_dists(i, j));
        prob_sum += r.first;
        gradient(i, j) = gradient(j, i) = r.second;
      }

// debug
  if (step == 0 || prob_sum < 1e-4) {
    log.printf("step %lu\n",step);
    for (size_t i = 0; i < size; ++i)
      for (size_t j = 0; j < i; ++j)
        if (dist_pairs[i][j]) {
//          real_dists(i, j) = real_dists(j, i) = delta(positions[i], positions[j]).modulo();
          log.printf("dist(%lu,%lu) = %f\n",i,j,real_dists(i,j));
	}
  
//    if (step > 0) abort();
  }
  step++;
// end debug

  for ( size_t i = 0; i < size; ++i) {
    Vector derivatives;
    for ( size_t j = 0; j < size; ++j) 
      if (dist_pairs[i][j])
        derivatives += delta(positions[j], positions[i]) * (gradient(i, j) / real_dists(i, j));

    setAtomsDerivatives(i, derivatives);
  }
  setBoxDerivativesNoPbc();
  setValue(prob_sum);


}

} // end of namespace afed
} // end of namespace colvar
} // end of namespace PLMD
