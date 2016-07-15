// co-occurrence -> loglikelihood matrix

// std libs

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>

// contrib library headers

#include <boost/program_options.hpp>

// local headers

#include "feature_map.hpp"
#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"



// most of this logic derived from Mahout logl Java implementation
// for Dunning style log likelihood ratio computation

// x log x
static inline double xlogx(double x) {
  return x == 0 ? 0.0 : x * log(x);
}

// Shannon entropy...
static inline double entropy2(double a, double b) {
  return xlogx(a + b) - xlogx(a) - xlogx(b);
}
  

static inline double entropy4(double a, double b, double c, double d) {
  return xlogx(a + b + c + d) - xlogx(a) - xlogx(b) - xlogx(c) - xlogx(d);
}

static inline double log_likelihood_ratio(double k11, double k12, double k21, double k22) {
  double r_entropy = entropy2(k11 + k12, k21 + k22);
  double c_entropy = entropy2(k11 + k21, k12 + k22);
  double m_entropy = entropy4(k11, k12, k21, k22);
  if (r_entropy + c_entropy < m_entropy) return 0;
  else return 2.0 * (r_entropy + c_entropy - m_entropy);
}



/////////////////////////////
// logl feature cooc matrix
/////////////////////////////

std::unique_ptr<sparse_matrix> logl_matrix(const sparse_matrix& S,
                                           const vector& feature_totals,
                                           const double llr_eps) {
  // total all features
  scalar_t te = feature_totals.sum();
  // accumulate ijvs for logl_matrix based on S and totals
  triplet_vec ijvs;
  ijvs.reserve(S.cols());

  // keep track of max logl
  double max_llr = 0;
  double min_llr = 10E20;
  
  // foreach feature -> feature occurrence count witnessed by
  // we can iterate over the CSC matrix where s[i,j] != 0

  for (int i = 0; i < S.cols(); ++i) {
    
    sparse_vector u = S.col(i); // column vector

    for (sparse_vector::InnerIterator it(u); it; ++it) {
      int j = it.index();
      double ab = it.value();

      // compute contingency table as in Dunning
      double k_11 = ab; // A(i) and B(j) together
      double k_12 = feature_totals[j] - ab;   // B but not A
      double k_21 = feature_totals[i] - ab;   // A but not B
      double k_22 = te - (feature_totals[i] + feature_totals[j]); // neither 

      // compute scaled loglr
      double llr = te * log_likelihood_ratio(k_11, k_12, k_21, k_22);
      // update max llr
      if (llr > max_llr) max_llr = llr;
      if (llr < min_llr) min_llr = llr;
      // help sparsity of L and only save if greater than lower bound llr_eps
      if (llr > llr_eps) ijvs.push_back(triplet(j, i, llr));
    }
  }

  std::cerr << "llr: [" << min_llr << "," << max_llr << "]" << std::endl;
  
  // logl matrix will be same shape as input cooc-matrix
  std::unique_ptr<sparse_matrix> L(new sparse_matrix(S.cols(), S.cols()));
  // populate sparse matrix
  L->setFromTriplets(ijvs.begin(), ijvs.end());
  L->makeCompressed();
  return L;
}



//////////////////////////////////////////////////
// entry point - process command args and options
//

int main(int argc, char** argv) {

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  
  desc.add_options()
    ("help", "compute loglikelihood matrix from coocurence matrix using stdio")

    ("logl_eps", po::value<double>()->default_value(1),
     "lower bound of logl in matrix to assist sparsity");



  // parse command line
  
  po::variables_map opts;
  
  po::store(po::parse_command_line(argc, argv, desc), opts);
  po::notify(opts);

  
  if (opts.count("help")) {
    std::cerr << desc <<  std::endl;
    return 1;
    
  } else {

    double logl_eps = opts["logl_eps"].as<double>();
    // compute logl of co-occurence and output both matrices
    
    std::cerr << "deserialise (1st order) co-occurrence matrix from stdin..." << std::endl;

    sparse_matrix A;
    deserialize_matrix(A);
   
    std::cerr << "density: " << 100 * ((double) A.nonZeros() / A.size()) << "%" << std::endl; 

    // sparse columnwise sum: matrix * 1-vector
    vector ones = vector::Ones(A.cols());

    // gets around adding transpose with wrong storage order...
    vector feature_totals = ((A) * ones) + (A.transpose() * ones);

    // compute logl matrix
    std::unique_ptr<sparse_matrix> L = logl_matrix(A, feature_totals, logl_eps);
    
    std::cerr << "save logL matrix: " << L->nonZeros()
              << " density: " << 100 * ((double) L->nonZeros() / L->size()) << "%" << std::endl; 

    // write out to stdout
    return (serialize_matrix(*L)) ? 0 : 1;
    
  }
}

