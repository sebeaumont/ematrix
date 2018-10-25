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


//////////////////////////////////////
// sparse matrix from filtered input
//////////////////////////////////////

std::unique_ptr<sparse_matrix> hipass_filter(sparse_matrix& C, sparse_matrix& L, const double logl) {
  
  // iterate L and create new matrix F from C where L exceeds threshold

  triplet_vec ijvs;
  ijvs.reserve(L.cols());
  
  for (int j = 0; j < L.outerSize(); ++j) {
    
    for (sparse_matrix::InnerIterator it(L, j); it; ++it) {
      int i = it.index();      // row index
      double llr = it.value(); // value
      // if logl threhold reached add source matrix value to output
      if (llr > logl) ijvs.push_back(triplet(i, j, C.coeffRef(i,j)));
    }
  }
  
  // create and populate sparse matrix  
  std::unique_ptr<sparse_matrix> F(new sparse_matrix(L.cols(), L.cols()));
  F->setFromTriplets(ijvs.begin(), ijvs.end());
  F->makeCompressed();
  return F;
}

//////////////////////////////////////////////////
// entry point - process command args and options
//

int main(int argc, char** argv) {

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  
  desc.add_options()
    ("help", "filter co-occurrence matrix using logl threshold and logl matrix")

    ("cooc", po::value<std::string>(),
     "filename of co-occurrence matrix")

    ("logl", po::value<double>(),
     "lower bound of logl to use as filter mask");


  // parse command line
  
  po::variables_map opts;
  
  po::store(po::parse_command_line(argc, argv, desc), opts);
  po::notify(opts);

  
  if (opts.count("help")) {
    std::cerr << desc <<  std::endl;
    return 1;

  } else if (!opts.count("cooc") || !opts.count("logl")) {
    std::cerr << "both cooc matrix file and logl threshold are required parameters!" << std::endl;
    return 1;
    
  } else {

    double logl = opts["logl"].as<double>();
    std::string cmatrix = opts["cooc"].as<std::string>();

    std::cerr << "deserialize co-ocurrence matrix from: " << cmatrix << "..." << std::endl;
    std::ifstream ins;
    ins.open(cmatrix);
    
    if (ins.good()) {
      sparse_matrix C;
      if (deserialize_matrix(C, ins)) {

        std::cerr << "deserialize logl matrix from stdin..." << std::endl;
        sparse_matrix L;
        
        if (deserialize_matrix(L)) {   
          std::cerr << "density: " << 100 * ((double) L.nonZeros() / L.size()) << "%" << std::endl;
   
          // filter C using L and logl lower bound
          std::unique_ptr<sparse_matrix> F = hipass_filter(C, L, logl);
          // write to stdout (tty?)
          std::cerr << "density: " << 100 * ((double) F->nonZeros() / F->size()) << "%" << std::endl;
          if (serialize_matrix(*F)) {
            return 1;
            
          } else std::cerr << "failed to serialise output matrix!" << std::endl;
          
        } else std::cerr << "failed to deserialise logl matrix!" << std::endl;
        
      } else std::cerr << "failed to deserialise cooc matrix: " << cmatrix << "!" << std::endl;
      
    } else std::cerr << "cannot read input cooc matrix: " << cmatrix << "!" << std::endl;
    return 0;
  }
}

