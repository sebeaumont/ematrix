# python module to sparse matrix loglg

import sys
import numpy as np
import scipy.sparse as sp
from itertools import combinations
from math import log

# utility object

class featuremap:

    "Object to keep track of map state"
    
    def __init__(self):
        self._map = {}
        self._index = 0 # N.B. zero based arrays

    def ensure_feature(self, name):
        fid = self._map.get(name, None)
        if fid:
            return fid
        else:
            self._map[name] = self._index
            self._index += 1
            return self._map[name]

    def size(self):
        return self._index

    

def reify_frame(index_list, frame_indexes):
    # extend list with reified symmetic relation
    feature_v = sorted(frame_indexes)
    index_list.extend([ij for ij in combinations(feature_v, 2)])
    feature_v.reverse()
    index_list.extend([ij for ij in combinations(feature_v, 2)])


############
# Step 1.
#

def samples2cooc(ins=sys.stdin):
    "Take a stream of tsv samples on ins and return featuremap and cooc matrix."
    # frame\tfeature\tvalue

    start = True
    last_frame = None
    features = featuremap()

    # accumulated i,j pairs
    ij_pairs = []
    
    # recycled list of indexes (features) in frame
    frame_indexes = []
    
    
    for line in ins:
        # N.B. we ignore values and assume 1
        frame, feature, _ = line.strip().split('\t')

        # get feature index
        fi = features.ensure_feature(feature)

        if start:
            last_frame = frame
            start = False
            
        elif frame != last_frame:
            # Done with frame...
            # N.B. create combinations of coocurrences to add to i,j,v arrays
            reify_frame(ij_pairs, frame_indexes)
            # clear for next frame
            frame_indexes = []

        # accumulate
        last_frame = frame
        frame_indexes.append(fi)
        
    # final output at end of input 
    reify_frame(ij_pairs, frame_indexes)
    
    print("cooc:", len(ij_pairs), "features:", features.size())
    
    # create a sparse IJV (coordinate) matrix to return. N.B. dups get summed over
    C = sp.coo_matrix(([1]*len(ij_pairs), ([i for i, j in ij_pairs], [j for i, j in ij_pairs])),
                      shape=(features.size(), features.size()), dtype=np.double)
    
    # should be all we need now
    return C, features._map



# LOGL computation
# most of this logic derived from Mahout logl Java implementation
# for Dunning style log likelihood ratio computation


def xlog(x):
    return 0.0 if x == 0 else x * log(x)  # log 0 -> 0

# Shannon entropy...
def entropy2(a, b):
    return xlog(a + b) - xlog(a) - xlog(b)

def entropy4(a, b, c, d):
    return xlog(a + b + c + d) - xlog(a) - xlog(b) - xlog(c) - xlog(d)


def log_likelihood_ratio(k11, k12, k21, k22):
    r_entropy = entropy2(k11 + k12, k21 + k22)
    c_entropy = entropy2(k11 + k21, k12 + k22)
    m_entropy = entropy4(k11, k12, k21, k22)
  
    if (r_entropy + c_entropy < m_entropy):
        return 0
    else:
        return 2.0 * (r_entropy + c_entropy - m_entropy)

    
###########
# Step 2.
#

def cooc2logl(C):
    "compute logl matrix from co-occurence matrix"

    # 1. need column/feature totals
    totals = C.sum(axis=0) # weird: but this is a degenate matrix with one row!
    occurs = totals.sum()
    
    # accumulate i,j,llr
    triples = []
    
    # N.B. speedy iteration over sparse entries
    for i, j, v in zip(C.row, C.col, C.data):
        
        k_11 = v                                 # occurrences of A (i) and B (j) together
        k_12 = totals[0,j] - v                   # B but not A
        k_21 = totals[0,i] - v                   # A but not B
        k_22 = occurs - (totals[0,i] + totals[0,j]) # neither A or B

        #print(i, j, "->", k_11, k_12, k_21, k_22)

        # compute scaled loglr
        llr = occurs * log_likelihood_ratio(k_11, k_12, k_21, k_22)
        # save i,j,v for llr
        triples.append((i, j, llr))


    # make the L matrix
    L = sp.coo_matrix(([l for i, j, l in triples], ([i for i, j, l in triples], [j for i, j, l in triples])),
                      dtype=np.double,
                      shape=C.shape)
    return L