#include <iostream>

#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"

using namespace std;

int main(int argc, char *argv[]) {

  sparse_matrix B;
  
  // read matrix from stdin
  if (deserialize_matrix(B)) {
  
    cerr << B.nonZeros() << endl;   
    cerr << B.size() << endl;
    return 1;
    
  } else {
    cerr << "couldn't deserialize matrix on stdin!" << endl;
    return 0;
  }
}
