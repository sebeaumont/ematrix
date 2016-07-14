// show sparse matrix

#include <iostream>
#include <fstream>

#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"

using namespace std;


void show_matrix(const sparse_matrix& M, ostream& outs=cerr) {
  
    outs << M.nonZeros() << endl;   
    outs << M.size() << endl;
    outs << 100 * ((float) M.nonZeros() / M.size()) << endl;

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
    outs << min_v << endl;
    outs << max_v << endl;
}
    


int main(int argc, char *argv[]) {

  sparse_matrix B;

  // filename on command line
  if (argc > 1) {
    
    string filename(argv[1]);
    std::ifstream ins;
    ins.open(filename);

    if (ins.good()) {

      cerr << "reading matrix data from: " << filename << endl;
      if (deserialize_matrix(B, ins)) show_matrix(B);
      else cerr << "cannot deserialize matrix!" << endl;
      
    } else cerr << "cannot read file: " << filename << "!" << endl;
    

  } else {
    cerr << "reading matrix data from stdin..." << endl;
    // read matrix from stdin
    if (deserialize_matrix(B)) show_matrix(B);
    else cerr << "couldn't deserialize matrix on stdin!" << endl;
  }
  
  return 0;
}
