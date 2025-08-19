#include "SparseMatrixCSR.h"
#include <iostream>
#include <cassert>


void testSingleElementMatrix() {
    std::cout << "Testing single element matrix..." << '\n';
    
    std::vector<std::vector<double>> dense = {{8.0}};
    
    SparseMatrixCSR sparse(1, 1);
    sparse.buildFromDense(dense);
    
    std::vector<double> vec = {2.0};
    std::vector<double> result = sparse.multiply(vec);
    
    assert(result[0] == 16.0);  // 8.0 * 2.0
    assert(sparse.getNNZ() == 1);
    
    std::cout << "Single element test passed" << '\n';
}

void testBasicMultiplication() {
    std::cout << "Testing basic multiplication..." << '\n';
    
    std::vector<std::vector<double>> dense = {
        {1.0, 0.0, 2.0},
        {0.0, 3.0, 0.0}, 
        {4.0, 0.0, 5.0}
    };
    
    SparseMatrixCSR sparse(3, 3);
    sparse.buildFromDense(dense);
    
    std::vector<double> vec = {1.0, 1.0, 1.0};
    std::vector<double> result = sparse.multiply(vec);
    
    assert(result[0] == 3.0);
    assert(result[1] == 3.0);
    assert(result[2] == 9.0);
    
    std::cout << "Basic multiplication passed" << '\n';
}

void testIdentityMatrix() {
    std::cout << "Testing identity matrix..." << '\n';
    
    std::vector<std::vector<double>> dense = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    SparseMatrixCSR sparse(3, 3);
    sparse.buildFromDense(dense);
    
    std::vector<double> vec = {7.0, 8.0, 9.0};
    std::vector<double> result = sparse.multiply(vec);
    

    assert(result[0] == 7.0);
    assert(result[1] == 8.0);
    assert(result[2] == 9.0);
    assert(sparse.getNNZ() == 3);
    
    std::cout << "Identity matrix test passed" << '\n';
}

void testEmptyMatrix() {
    std::cout << "Testing empty matrix..." << '\n';
    
    std::vector<std::vector<double>> dense = {
        {0.0, 0.0},
        {0.0, 0.0}
    };
    
    SparseMatrixCSR sparse(2, 2);
    sparse.buildFromDense(dense);
    
    assert(sparse.getNNZ() == 0);
    std::cout << "Empty matrix test passed" << '\n';
}

void testDimensionMismatch() {
    std::cout << "Testing dimension mismatch..." << '\n';
    
    std::vector<std::vector<double>> dense = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    SparseMatrixCSR sparse(2, 2);
    sparse.buildFromDense(dense);
    
    std::vector<double> wrong_size_vec = {1.0, 2.0, 3.0}; // should be size 2 not 3
    
    try {
        sparse.multiply(wrong_size_vec);
        assert(false);
    } catch (const std::invalid_argument& e) {
        std::cout << "Dimension mismatch correctly caught" << '\n';
    }
}

void testBasicTranspose() {
    std::cout << "Testing basic transpose..." << '\n';
    
    // original: [[1, 2], [3, 0]]
    // should be: [[1, 3], [2, 0]]
    std::vector<std::vector<double>> dense = {
        {1.0, 2.0},
        {3.0, 0.0}
    };
    
    SparseMatrixCSR sparse(2, 2);
    sparse.buildFromDense(dense);
    
    SparseMatrixCSR transposed = sparse.transpose();
    
    // verify dimensions are swapped
    assert(transposed.getRows() == 2);
    assert(transposed.getCols() == 2);
    assert(transposed.getNNZ() == 3);
    

    std::vector<double> vec1 = {1.0, 1.0};
    std::vector<double> result1 = transposed.multiply(vec1);
    assert(result1[0] == 4.0);  
    assert(result1[1] == 2.0);  
    
    std::cout << "Basic transpose passed" << '\n';
}

void testTransposeIdentity() {
    std::cout << "Testing transpose of identity matrix..." << '\n';
    
    std::vector<std::vector<double>> dense = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    
    SparseMatrixCSR sparse(3, 3);
    sparse.buildFromDense(dense);
    
    SparseMatrixCSR transposed = sparse.transpose();
    
    // identity matrix should be its own transpose
    std::vector<double> vec = {1.0, 2.0, 3.0};
    std::vector<double> original_result = sparse.multiply(vec);
    std::vector<double> transposed_result = transposed.multiply(vec);
    
    for(int i = 0; i < 3; ++i) {
        assert(original_result[i] == transposed_result[i]);
    }
    
    std::cout << "Identity transpose passed" << '\n';
}

void testTransposeRectangular() {
    std::cout << "Testing rectangular matrix transpose..." << '\n';
    

    std::vector<std::vector<double>> dense = {
        {1.0, 0.0, 2.0},
        {3.0, 4.0, 0.0}
    };
    
    SparseMatrixCSR sparse(2, 3);
    sparse.buildFromDense(dense);
    
    SparseMatrixCSR transposed = sparse.transpose();
    
    assert(transposed.getRows() == 3); 
    assert(transposed.getCols() == 2);
    assert(transposed.getNNZ() == 4);
    
    std::cout << "Rectangular transpose passed" << '\n';
}

int main() {
    testSingleElementMatrix();
    testBasicMultiplication();
    testIdentityMatrix();
    testEmptyMatrix();
    testDimensionMismatch();
    testBasicTranspose();
    testTransposeIdentity();
    testTransposeRectangular();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
