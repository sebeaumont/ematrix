// feature space, co-occurrence, logliklihood filtering

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <cmath>
#include <boost/program_options.hpp>
#include <Eigen/SparseCore>

#include "sparse_matrix_io.hpp"

/////////////////////////////////////////////////////////////////////////
// types we need for a sparse co-occurrence matrix and similarity scores
//

typedef float element_t;
typedef Eigen::SparseMatrix<element_t> sparse_matrix;
typedef Eigen::SparseVector<element_t> sparse_vector;
typedef Eigen::Triplet<element_t>      triplet;
typedef std::vector<triplet>           triplet_vec;
typedef Eigen::VectorXf                vector;

/////////////////////////////////////
// custom bimap with size_t indexes 

template <typename L>
class index_bimap {
  std::map<L, std::size_t> _left;
  std::map<std::size_t, L> _right;
  
public:
  
  inline void insert(const L& l, const std::size_t& r) {
    _left[l] = r;
    _right[r] = l;
  }
  
  inline const std::size_t left(const L& l) const { return _left.at(l); }
  
  inline const L right(const std::size_t& r) const { return _right.at(r); }
  
  inline const std::size_t ensure(const L& l) {
    try {
      return _left.at(l);
    } catch (std::out_of_range& ex) {
      std::size_t i = _left.size();
      _left[l] = i;
      _right[i] = l;
      return i;
    }
  }
};


////////////////////////////////////////////
// map types for key aggregation/reduction

typedef index_bimap<std::string> index_map;


////////////////////////////////////////////
// function objects for aggregation binops

struct max { 
  float operator()(float x, float y) const { return x > y ? x : y; }; 
};

struct min { 
  float operator()(float x, float y) const { return x < y && x ? x : y; }; 
};

struct sum { 
  float operator()(float x, float y) const { return x + y; }; 
};

struct avg {
  float operator()(float x, float y) const { return (x + y) / 2.0; }; 
};

struct cnt {
  // N.B. assumes x, y are in [0,1]
  float operator()(float x, float y) const { return ceil(x) + ceil(y); };
};


//////////////////////////////////////
// i,j,v, csc based sparse matrix ops
//////////////////////////////////////


int inline add_pairwise_triplets(const std::vector<int>& indexes, triplet_vec& triplets) {
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
                                                       index_map& fmap,
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
  float value;
  
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
  std::cerr << "samples:\t" << samples << std::endl
            << "max index:\t" << max_index << std::endl
            << "population:\t" << population << std::endl;

  // create the sparse matrix from final list of (i,j,v) triplets
  // -- by default multiple i,j pairs are summed over
  std::unique_ptr<sparse_matrix> M(new sparse_matrix(max_index+1, max_index+1));
  M->setFromTriplets(triplets.begin(), triplets.end());

  std::cerr << "pairs:\t" << M->nonZeros() << std::endl;
  M->makeCompressed(); // csc?
  return M;
}


/////////////////////////////////////////////
// incrementally build up the triplet vector
//

typedef std::pair<int,float> feature_pair;
typedef std::vector<feature_pair> feature_values_t;

inline void sample_triplets(const int sample_id, const feature_values_t& feature_indexes, triplet_vec& triplets) {
  for (auto fix = feature_indexes.begin(); fix != feature_indexes.end(); ++fix) {
    triplets.push_back(triplet(fix->first, sample_id, fix->second));
  } 
}


// aggregation binop 

typedef boost::function<float (float, float)> aggfn_t;


//////////////////////////////////////////////////
// create sample matrix of sparse feature vectors

std::unique_ptr<sparse_matrix> feature_matrix_from_stream(std::istream& in, index_map& features, index_map& samples,
                                                          aggfn_t& aggfn, const int population_guess = (10 * 1024 * 1024)) {
  
  // list of triplets (i,j,v) i is sample index, j is sparse feature index
  triplet_vec triplets;
  triplets.reserve(population_guess);

  // accumulators
  std::size_t max_index = 0;
  std::size_t max_sample_index = 0;
  std::size_t n_samples = 0;
  
  // state
  bool first = true;
  std::string last_sample;
  feature_values_t feature_indexes;

  // read samples and feature tsv list from in stream
  // N.B. it is assumed that input is grouped (sorted input) by sample; else we would have
  // to keep a map of index vectors by frame.
  
  std::string sample;
  std::string feature;
  float value;
  
  while (in >> sample >> feature >> value) {

    // get feature index 
    std::size_t index = features.ensure(feature);
    max_index = index > max_index ? index : max_index;      
    ++n_samples;

    // accumulate sample features
    
    if (first) {
      last_sample = sample;
      first = false;

    } else if (sample != last_sample) {
      // done this sample -- emit vector
      // get sample index
      std::size_t sample_index = samples.ensure(last_sample);
      max_sample_index = sample_index > max_sample_index ? sample_index : max_sample_index;      
      sample_triplets(sample_index, feature_indexes, triplets);
      feature_indexes.clear();
    }
    
    feature_indexes.push_back(feature_pair(index, value));
    last_sample = sample;    
  }
  
  // tail
  std::size_t sample_index = samples.ensure(last_sample);
  max_sample_index = sample_index > max_sample_index ? sample_index : max_sample_index;      
  sample_triplets(sample_index, feature_indexes, triplets);


  // sample stats
  std::cerr << "individuals:\t" << max_sample_index+1 << std::endl
            << "features:\t" << max_index+1 << std::endl
            << "samples:\t" << n_samples << std::endl;

  // create the matrix
  std::unique_ptr<sparse_matrix> M(new sparse_matrix(max_index+1, max_sample_index+1));
  
  // XXX implicit reduction is sum if we use setFromTriplets...
  // XXX valued (as opposed to binary features) will require explicit reduction functions

  // apply a specific reduction function instead of default (max, min, sum, avg...)
  // M->setFromTriplets(triplets.begin(), triplets.end());
  for (auto ti = triplets.begin(); ti != triplets.end(); ++ti) {
    const element_t v = M->coeffRef(ti->row(), ti->col());
    M->coeffRef(ti->row(), ti->col()) = aggfn(v, ti->value());
  }
  
  M->makeCompressed();
  return M;
}


//////////////////////
// cosine similarity
/////////////////////

inline const double cosine_similarity(const sparse_vector& u, const sparse_vector& v, const int n) {
  
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

void cosine_scores(std::unique_ptr<sparse_matrix>& A, std::unique_ptr<sparse_matrix>& B,
                   const index_map& fidx, std::ostream& outs) {

  // convert upper triangular forms to symmetric matrix so we can efficiently extract columns
  
  //  1. get transposes -- this is to overcome eigen just changing the
  //  storage order in transpose which is a good trick but loses here
  //  as we need same storage order to add matrices
  
  sparse_matrix Y = sparse_matrix(A->transpose()) + *A;
  sparse_matrix X = sparse_matrix(B->transpose()) + *B;

  // we allow comparison of spaces of different sizes knowing that indexes are comparable
  // we take dot products on the shorter vectors and norms w.r.t to both  
 
  const int n = X.cols() <= Y.cols() ? X.cols() : Y.cols(); // min space dim   
  const int d = X.cols() > Y.cols() ? X.cols() : Y.cols();  // max space dim
  
  // 3. extract columns and compute cosine similarities feature by feature
  // -- N.B. this is embarassingly parallel

  for (int i = 0; i < n; ++i) {
    
    sparse_vector u = X.col(i);
    sparse_vector v = Y.col(i);
    
    const double cs = cosine_similarity(u, v, n);
    
    // output feature score and counts
    outs << fidx.right(i) << "\t" << cs << "\t"  << (int) v.sum() << "\t" << (int) u.sum() << std::endl;
  }

  // 4. output remainder of largest space
  
  for (int i = n; i < d; ++i) {
    
    double usum = X.cols() >= Y.cols() ? X.col(i).sum() : 0.0;
    double vsum = X.cols() < Y.cols() ? Y.col(i).sum() : 0.0;
    
    outs << fidx.right(i) << "\t" << 0.0 << "\t"  << (int) vsum << "\t" << (int) usum << std::endl;
  }
}


///////////////////////////
// output feature vectors

void feature_vectors(std::unique_ptr<sparse_matrix>& S, const index_map& features, const index_map& samples, std::ostream& outs) {

  const int n_samples = S->cols();
  
  for (int i = 0; i < n_samples; ++i) {
    outs << samples.right(i);

    // N.B. must be column (default) storage order 
    sparse_vector f = S->col(i);
    
    for (sparse_vector::InnerIterator it(f); it; ++it) {
      outs << "\t" << features.right(it.index()) << ":" << it.value();
    }
    outs << std::endl;
  }
}

/////////////////////////////////////////////////////////////////
// output sparse matrix of pairwise feature values as a tsv list

void feature_matrix(std::unique_ptr<sparse_matrix>& S, const index_map& features, std::ostream& outs) {

  const int n = S->cols();
  
  for (int i = 0; i < n; ++i) {

    // N.B. must be column (default) storage order 
    sparse_vector f = S->col(i);
    
    for (sparse_vector::InnerIterator it(f); it; ++it) {
      outs << it.value() << "\t" << features.right(i) << "\t" << features.right(it.index()) << std::endl;
    }
  }
}


// most of this logic derived from Mahout logl Java implementation

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

std::unique_ptr<sparse_matrix> logl_matrix(std::unique_ptr<sparse_matrix>& S, const vector& feature_totals) {
  // total all features
  float te = feature_totals.sum();
  // accumulate ijvs for logl_matrix based on S and totals
  triplet_vec ijvs;
  ijvs.reserve(S->cols());

  // keep track of max logl
  double max_llr = 0;
  
  // foreach feature -> feature occurrence count witnessed by
  // we can iterate over the CSC matrix where s[i,j] != 0

  for (int i = 0; i < S->cols(); ++i) {
    
    sparse_vector u = S->col(i); // column vector

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
      ijvs.push_back(triplet(j, i, llr));
      if (llr > max_llr) max_llr = llr;
    }
  }

  std::cerr << "max llr: " << max_llr << std::endl;
  
  // logl matrix will be same shape as input cooc-matrix
  std::unique_ptr<sparse_matrix> L(new sparse_matrix(S->cols(), S->cols()));
  // populate sparse matrix
  L->setFromTriplets(ijvs.begin(), ijvs.end());
  L->makeCompressed();
  return L;
}

////////////////////////////////////////////
// sparse binary matrix from filtered input
////////////////////////////////////////////

std::unique_ptr<sparse_matrix> hipass_filter(std::unique_ptr<sparse_matrix>& C, std::unique_ptr<sparse_matrix>& L, double logl) {
  // iterate L and create new matrix F from C where L exceeds threshold

  triplet_vec ijvs;
  ijvs.reserve(L->cols());
  
  for (int j = 0; j < L->cols(); ++j) {
    
    sparse_vector u = L->col(j); // column vector

    for (sparse_vector::InnerIterator it(u); it; ++it) {
      int i = it.index();      // row index
      double llr = it.value(); // value

      if (llr > logl)
        ijvs.push_back(triplet(i, j, C->coeffRef(i,j)));
    }
  }
  
  // create and populate sparse matrix  
  std::unique_ptr<sparse_matrix> F(new sparse_matrix(L->cols(), L->cols()));
  F->setFromTriplets(ijvs.begin(), ijvs.end());
  F->makeCompressed();
  return F;
}

////////////////////////////////////////
// quick and dirty file existence check 

inline bool file_exists(const char *path) {
  return std::ifstream(path).good();
}

inline bool file_exists(std::string& path) {
  return std::ifstream(path).good();
}


/////////////////////////////////////////
//// turn user input into an aggfn object

// xxx could create a map for this...
aggfn_t name_to_aggfn(const std::string& s) {
  if (s == "min") return min();
  else if (s == "sum") return sum();
  else if (s == "avg") return avg();
  else if (s == "cnt") return cnt();
  else return max();
}


//////////////////////////////////////////////////
// entry point - process command args and options
//

int main(int argc, char** argv) {

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  po::positional_options_description p;
  p.add("reference", 1);

  desc.add_options()
    ("help", "Feature space analysis utility\n\
              \tinput samples are read from stdin \n\
              \tif no comparison data is supplied then output feature vectors for samples\n\
              \tusing aggfunc to reduce feature occurrence values in sample, feature, value input.")
    ("aggfunc", po::value<std::string>()->default_value("max"),
     "feature aggregation operator: one of: max, min, sum, avg, cnt <max>")
    ("logl", po::value<float>(),
     "compute log likelihood of 1st order co-occurrence matrix and use the given threshold\
      to filter statistically significant co-occurrences")
    ("reference", po::value<std::string>(), "file of reference observations (should be a valid path)");

  // announce
  std::cerr << "feature space model analysis  " << std::endl;
  std::cerr << "============================" << std::endl;

  po::variables_map opts;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), opts);
  po::notify(opts);

  // updated dynamically from input data to do rollup
  index_map features;
  
  if (opts.count("help")) {
    std::cerr << desc <<  std::endl;
    return 1;
    
  } else if (opts.count("logl")) {

    float logl = opts["logl"].as<float>();
    // compute logl of co-occurence and output both matrices
    
    std::cerr << "creating pairwise (1st order) co-occurrence matrix from data..." << std::endl;
    std::unique_ptr<sparse_matrix> A = cooc_matrix_from_stream(std::cin, features);

    std::cerr << A->nonZeros() << "/" << A->size() << std::endl; 

    // sparse columnwise sum: matrix * 1-vector
    vector ones = vector::Ones(A->cols());

    // gets around adding transpose with wrong storage order...
    vector feature_totals = ((*A) * ones) + (A->transpose() * ones);

    // compute logl matrix
    std::unique_ptr<sparse_matrix> L = logl_matrix(A, feature_totals);

    std::cerr << "logL entries: " << L->nonZeros() << std::endl;
    
    if (logl > 0) {
      std::cerr << "output filter with logL-threshold: " << logl << std::endl;
      // compute filtered co-occurrence matrix
      std::unique_ptr<sparse_matrix> F = hipass_filter(L, A, logl);
      std::cerr << "filtered co-oc: " << F->nonZeros() << std::endl;
      // UC: save matrix
      serialize(*F);
      // render filtered matrix
      feature_matrix(F, features, std::cout);
      
    } else {
      std::cerr << "output logl matrix: " << logl << std::endl;
      feature_matrix(L, features, std::cout);
    }
    
    
  } else if (!opts.count("reference")) {
    
    // no reference data given: so we create and output feature matrix

    // select an aggregation function for feature values
    std::string aggfunc = opts["aggfunc"].as<std::string>(); 
    aggfn_t fn = name_to_aggfn(aggfunc);
    
    // sample index
    index_map samples;
    
    std::cerr << "create feature vectors: aggregation fn:" << aggfunc << std::endl;
    
    std::unique_ptr<sparse_matrix> S = feature_matrix_from_stream(std::cin, features, samples, fn);

    std::cerr << "---------------------------------------------------------" << std::endl;

    feature_vectors(S, features, samples, std::cout);
    
  } else {

    std::string rpath(opts["reference"].as<std::string>());

    // check files exist
    if (!file_exists(rpath)) {
      std::cerr << "file path: " << rpath << " does not exist!" << std::endl;
      return 5;
    }

    // get file open before we start the heavy lifting
    std::ifstream ins1(rpath, std::ios::in);

    ///////////////////////////////////////////////////////////////////////
    // create the sparse upper triangular feature co-occurrence matrices
    //////////////////////////////////////////////////////////////////////
    
    std::cerr << "creating co-occurrence matrix for reference data: " << rpath << "..." << std::endl;
    std::unique_ptr<sparse_matrix> A = cooc_matrix_from_stream(ins1, features);
    std::cerr << "---------------------------------------------------------" << std::endl; 

    std::cerr << "creating co-occurrence matrix for sample data on stdin... " << std::endl;
    std::unique_ptr<sparse_matrix> B = cooc_matrix_from_stream(std::cin, features);
    std::cerr << "---------------------------------------------------------" << std::endl; 


    ////////////////////////////////
    // compute cosine similarities
    ///////////////////////////////
    
    std::cerr << "computing similarity scores..." << std::endl; 

    cosine_scores(A, B, features, std::cout);
  }
  return 0;
}


/* Local Variables: */
/* compile-command: "(cd build; make)" */
/* End: */
