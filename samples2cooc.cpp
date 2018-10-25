// read sample data to create sparse co-occurrence matrix and feature map

// std libs

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>


// local headers

#include "feature_map.hpp"
#include "sparse_matrix.hpp"
#include "sparse_matrix_io.hpp"


#include <unistd.h>
#include <stdio.h>


//////////////////////////////////////
// i,j,v, csc based sparse matrix ops
//////////////////////////////////////


static inline int add_pairwise_triplets(const std::vector<int>& indexes, triplet_vec& triplets) {
  // create non-redundant pairwise associations for a set of indexes and add to triplet vector

  // 1. create ordered set of indexes
  std::set<int> idxset(indexes.begin(), indexes.end());
  int n = 0;

  // 2. create pairwise combinatations of indexes  
  for (auto first = idxset.begin(); first != idxset.end(); ++first) {
    for (auto next = std::next(first); next != idxset.end(); ++next) {
      // 3. add i,j,v triple to triplets
      triplets.push_back(triplet(*first, *next, 1));      
      ++n;
    }
  }  
  // return number of combinations i.e. binomial coefficient which in
  // this case is (choose 2 from idxset.size)
  return n;
}


/////////////////////////////////////////////////////////////////
// create a sparse co-occurrence matrix from input data stream:
// frame, feature, value
/////////////////////////////////////////////////////////////////

std::unique_ptr<sparse_matrix> cooc_matrix_from_stream(std::istream& in,
                                                       feature_map& fmap,
                                                       const int population_guess = (10 * 1024 * 1024)) {
  
  // accumulate list of triplets (i,j,v) for sparse matrix creation
  triplet_vec triplets;
  triplets.reserve(population_guess);

  // accumulators
  int max_index = 0;
  int population = 0;
  int samples = 0;
  
  // state
  bool first = true;
  std::string last_frame;
  std::vector<int> frame_indexes;

  // read frames and feature tsv list from in stream N.B. assumed that
  // input is sorted by frame else we would have to keep a map of
  // index vectors by frame.
  
  std::string frame;
  std::string feature;
  scalar_t value;
  
  while (in >> frame >> feature >> value) {

    // get feature index 
    std::size_t index = fmap.ensure(feature);
    
    ++samples;
    max_index = index > max_index ? index : max_index;

    // accumulate frame indexes then create combinations
    if (first) {
      last_frame = frame;
      first = false;

    } else if (frame != last_frame) {
      // frame change -- accumulate (i,j,v) for previous frames pariwise combinations of features (frame_indexes)
      // i.e. frames cause grouping and aggregation (sum) of feature counts in context 
      population += add_pairwise_triplets(frame_indexes, triplets);
      frame_indexes.clear(); // reset for new frame
    }
    
    frame_indexes.push_back(index);
    last_frame = frame;
  }

  // last frame
  population += add_pairwise_triplets(frame_indexes, triplets);

  // sample stats
  std::cerr << "samples:    " << samples << std::endl
            << "features:   " << max_index << std::endl
            << "co-occurs:  " << population << std::endl;

  // create the sparse matrix from final list of (i,j,v) triplets
  // -- by default multiple i,j pairs are summed over
  
  std::unique_ptr<sparse_matrix> M(new sparse_matrix(max_index+1, max_index+1));
  M->setFromTriplets(triplets.begin(), triplets.end());

  std::cerr << "distinct:   " << M->nonZeros() << std::endl;
  M->makeCompressed();
  return M;
}


//////////////////////////////////////////////////
// entry point - process command args and options
//

int main(int argc, char** argv) {

  if (argc != 2) {

    std::cerr << "Please provide a file name argument for output feature list!" << std::endl;   

  } else {

    std::string index_filename(argv[1]);
    feature_map features;

    // create new file for features
    std::fstream ofs;
    ofs.open(index_filename, std::fstream::out);
    if (ofs.good()) {
    
      std::cerr << "creating pairwise (1st order) co-occurrence matrix from data..." << std::endl;
      std::unique_ptr<sparse_matrix> A = cooc_matrix_from_stream(std::cin, features);

      std::cerr << "writing feature index to: " << index_filename << std::endl;
      features.serialize(ofs);
      ofs.close();

      std::cerr << "writing co-occurrence matrix to stdout..." << std::endl;
      
      // serialize cooc matrix to stdout if not tty xxx todo factor this into util header xxx
      if (!isatty(fileno(stdout))) {
        return (serialize_matrix(*A) ? 1 : 0);
      } else {
        std::cerr << "will not write binary matrix data to tty!" << std::endl;
        return 0;
      }
      
    } else {
      std::cerr << "cannot open file for writing!: " << index_filename << std::endl;
      return 0;
    }
  }
}
