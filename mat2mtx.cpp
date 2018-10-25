// convert eigen3 sparse matrix to matrix market format


#include <iostream>
#include <fstream>

#include <unsupported/Eigen/SparseExtra>

#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"


using namespace std;


void show_matrix(const sparse_matrix& M, ostream& outs=cerr) {
  
  outs << "nonz:\t" << M.nonZeros() << endl;   
  outs << "size:\t" << M.size() << endl;
  outs << "rho:\t" << 100 * ((float) M.nonZeros() / M.size()) << endl;

  scalar_t max_v = 0;
  scalar_t min_v = 10E20; //hmm need a max scalar_t!
    
  // iterate non-zero values to accumulate value stats
  for (int j = 0; j < M.outerSize(); ++j) {
    for (sparse_matrix::InnerIterator it(M, j); it; ++it) {
      //it.index();
      scalar_t v = it.value();
      max_v = (v > max_v) ? v : max_v;
      min_v = (v < min_v) ? v : min_v;
    }
  }   
  outs << "min:\t" << min_v << endl;
  outs << "max:\t" << max_v << endl;
}


int main(int argc, char *argv[]) {

  // filename on command line
  if (argc == 3) {
    
    string infile(argv[1]);
    string outfile(argv[2]);

    // open serialized eigen3 sparse matrix
    std::ifstream ins;
    ins.open(infile);

    if (ins.good()) {
      
      sparse_matrix B;
      cerr << "reading matrix data from: " << infile << endl;

      if (deserialize_matrix(B, ins)) {
        show_matrix(B);
        
        cerr << "writing matrix to: " << outfile << endl;
        Eigen::saveMarket(B, outfile); 
        // Eigen::saveMarket(B, outfile, Eigen::Symmetric); doesn't save space in output!!!
        cerr << "...done" << endl;
        
      } else cerr << "cannot deserialize input matrix file: " << infile << endl;
      
    } else cerr << "cannot read file: " << infile << endl;
    

  } else {
    cerr << "require both input matrix and output matrixmarket filenames" << endl; 
    return 1;
  }
  
  return 0;
}

