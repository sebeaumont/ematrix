#include <ctime>
#include <iostream>
#include <cstdlib>
#include <fstream>

#include "sparse_matrix_io.hpp"

using namespace std;

int main(int argc, char *argv[]){
    int rows, cols;
    rows = cols = 8;

    srand(time(0)); // use current time as seed for random generator
    
    typedef int element_t;
    typedef Eigen::Triplet<element_t> ijv_triplet_t;
    typedef Eigen::SparseMatrix<element_t> matrix_t;

    matrix_t A(rows,cols), B;

    std::vector<ijv_triplet_t> tv;

    tv.push_back(ijv_triplet_t(0, 0, rand()));
    tv.push_back(ijv_triplet_t(1, 1, rand()));
    tv.push_back(ijv_triplet_t(2, 2, rand()));
    tv.push_back(ijv_triplet_t(3, 3, rand()));
    tv.push_back(ijv_triplet_t(4, 4, rand()));
    tv.push_back(ijv_triplet_t(5, 5, rand()));
    tv.push_back(ijv_triplet_t(2, 4, rand()));
    tv.push_back(ijv_triplet_t(3, 1, rand()));
    tv.push_back(ijv_triplet_t(6, 7, rand()));

    // set matrix from i,j,v triplets
    A.setFromTriplets(tv.begin(), tv.end());
    cout << A.nonZeros() << endl;   
    cout << A.size() << endl;       
    cout << A << endl;              

    // open output file
    std::fstream outs;
    outs.open("matrix", std::ios::binary | std::ios::out);

    // write the matrix
    assert(serialize_matrix(A, outs));
    outs.close();
    
    // open input file
    std::fstream ins;
    ins.open("matrix", std::ios::binary | std::ios::in);

    // read matrix
    assert(deserialize_matrix(B, ins));
    ins.close();
    
    cout << B.nonZeros() << endl;   
    cout << B.size() << endl;       
    cout << B << endl;              

    // TODO equality assertion
    
    return 0;
}
