#include <iostream>

#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"

using namespace std;

int main(int argc, char *argv[]) {

  sparse_matrix B;
  
  // read matrix from stdin
  assert(deserialize(B));
  
  cout << B.nonZeros() << endl;   
  cout << B.size() << endl;       

  return 1;
}
