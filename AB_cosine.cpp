// co-occurrence + loglikelihood matrix + threshold -> filtered co-occurence matrix

// std libs

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


//////////////////////
// cosine similarity
/////////////////////

inline const double cosine_similarity(const sparse_vector& u, const sparse_vector& v, const int n) {

  // XXX we might not need this flexiblity anymore as the dimensions of the feature vectors should be the same
  // N.B. non-standard linear algebra: we allow vectors of differing
  // lengths to be compared by doing the dot product on the smaller
  // dimension and norms on the full length
  
  double dotp = u.head(n).dot(v.head(n));
  if (dotp == 0.0) return 0.0;

  double u_norm = u.norm();
  if (u_norm == 0.0) return 0.0;

  double v_norm = v.norm();
  if (v_norm == 0.0) return 0.0;

  double d = (dotp /  (u_norm * v_norm));
  
  return d;
}


///////////////////////////////////////////
// sparse feature matrix cosine comparison

void cosine_scores(sparse_matrix& A, sparse_matrix& B, std::ostream& outs=std::cout) {

  // TODO make this more friendly!
  assert(A.size() == B.size());

  const int n = A.cols();

  /*
  std::vector<double> scores;
  scores.reserve(n);
  */
  
  // iterate over column (feature) vectors
  // -- N.B. this is embarassingly parallel

  for (int i = 0; i < n; ++i) {
    
    sparse_vector u = A.col(i);
    sparse_vector v = B.col(i);

    const double cs = cosine_similarity(u, v, n);
    //    scores.push_back(cs);
    
    // output scores
    outs << cs << "\t"  << u.sum() << "\t" << v.sum() << std::endl;
  }
}



//////////////////////////////////////////////////
// entry point - process command args and options
//

int main(int argc, char** argv) {

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  
  desc.add_options()
    ("help", "Compare feature vectors in matrices using cosine simliarity measure")

    ("reference", po::value<std::string>(),
     "filename of A (reference) matrix");


  // parse command line
  
  po::variables_map opts;
  
  po::store(po::parse_command_line(argc, argv, desc), opts);
  po::notify(opts);

  
  if (opts.count("help")) {
    std::cerr << desc <<  std::endl;
    return 1;

  } else if (!opts.count("reference")) {
    std::cerr << "reference matrix is a required parameter!" << std::endl;
    return 1;
    
  } else {

    std::string amatrix = opts["reference"].as<std::string>();

    std::cerr << "deserialize (A) matrix from: " << amatrix << "..." << std::endl;
    std::ifstream ins;
    ins.open(amatrix);
    
    if (ins.good()) {
      sparse_matrix A;
      if (deserialize_matrix(A, ins)) {

        std::cerr << "deserialize (B) matrix from stdin..." << std::endl;
        sparse_matrix B;
        
        if (deserialize_matrix(B)) {   

          cosine_scores(A, B);
          
        } else std::cerr << "failed to deserialise (B) matrix!" << std::endl;
        
      } else std::cerr << "failed to deserialise (A) matrix: " << amatrix << "!" << std::endl;
      
    } else std::cerr << "cannot read input reference (A) matrix file: " << amatrix << "!" << std::endl;
    return 0;
  }
}

