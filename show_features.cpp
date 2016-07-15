// show sparse matrix

#include <iostream>
#include <fstream>

#include "feature_map.hpp"
#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"

using namespace std;

    
/////////////////////////////////////////////////////////////////
// output sparse matrix of pairwise feature values as a tsv list

void show_feature_matrix(const sparse_matrix& F,
                         const feature_map& features,
                         std::ostream& outs=cout) {

  // iterate non-zero values...
  for (int j = 0; j < F.outerSize(); ++j) {
    for (sparse_matrix::InnerIterator it(F, j); it; ++it) {
      outs << it.value() << "\t" << features.right(j) << "\t" << features.right(it.index()) << "\n";
    }
  }
}
  
int main(int argc, char *argv[]) {

  sparse_matrix B;
  feature_map features;
  
  // filename on command line
  if (argc > 1) {
    
    string filename(argv[1]);
    std::ifstream ins;
    ins.open(filename);

    if (ins.good()) {

      cerr << "reading feature index from: " << filename << endl;
      cerr << "reading matrix data from stdin..." << endl;
      
      if (features.deserialize(ins) && deserialize_matrix(B)) {
        ins.close();
        show_feature_matrix(B, features);
        return 0;
        
      } else cerr << "cannot deserialize!" << endl;
      
    } else cerr << "cannot read file: " << filename << "!" << endl;
    

  } else cerr << "required argument: feature index filename" << endl;
  return 1;
}
