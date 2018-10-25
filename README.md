# Sparse Matrix Representations of Feature Space

## Co-occurrence analysis - bread and butter metrics and filtering provides the following tools:
    
- samples2cooc  - create co-occurrence matrix and feature list from samples
- cooc2logl     - compute log likelihood of co-occurrences
- loglfilter    - filter input co-occurrence matrix based on logl threshold
- AB-cosine     - compare feature vectors in co-occurrence matrix
- show_features - human readable feature vectors using co-occurrence matrix and feature list
- dumpmat       - dump matrix contents to stdout useful for eyeballing computations and debugging


### TODO

- End to End verification and validation
- Scale test w.r.t to Isobel corpus

### DONE

- Co-occurrence from input triples
- Aggregation operators
- LogLiklihood filtering 
- Sparse matrix io module to (de-)serialize Eigen3 sparse matrices
- Split the monolith into stdio sparse matrix pipeline (New)
- (De-)serialize feature maps (New) 

