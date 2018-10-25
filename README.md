# Sparse Matrix Representations of Feature Space using Eigen3

## Co-occurrence analysis - bread and butter metrics and filtering provides the following tools:
    
- samples2cooc  - create co-occurrence matrix and feature list from samples
- cooc2logl     - compute log likelihood of co-occurrences
- loglfilter    - filter input co-occurrence matrix based on logl threshold
- AB-cosine     - compare feature vectors in co-occurrence matrix
- show_features - human readable feature vectors using co-occurrence
  matrix and feature list
  
## Also provides some general eigen3 matrix utilities

- dumpmat       - dump matrix contents to stdout useful for eyeballing
  computations and debugging
  
## And useful C++ headers

- index_bimap.hpp 
- sparse_matrix.hpp
- sparse_matrix_io.hpp



