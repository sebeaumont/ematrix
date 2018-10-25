// show sparse matrix

#include <iostream>
#include <fstream>

#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"

using namespace std;


void dump_matrix(const sparse_matrix& M, ostream& outs=cerr) {
  outs << M << endl;
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
      if (deserialize_matrix(B, ins)) dump_matrix(B);
      else cerr << "cannot deserialize matrix!" << endl;
      
    } else cerr << "cannot read file: " << filename << "!" << endl;
    

  } else {
    cerr << "reading matrix data from stdin..." << endl;
    // read matrix from stdin
    if (deserialize_matrix(B)) dump_matrix(B);
    else cerr << "couldn't deserialize matrix on stdin!" << endl;
  }
  
  return 0;
}
