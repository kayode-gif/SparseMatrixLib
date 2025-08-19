# Sparse Matrix Library

A C++ implementation of sparse matrices using the Compressed Sparse Row (CSR) format. I built this to learn about sparse matrix algorithms and performance optimization techniques.

## What it does

This library implements sparse matrix operations that are much more efficient than dense matrices when dealing with data that has lots of zeros. The CSR format stores only the non-zero elements, which can save massive amounts of memory and computation time.

### Key features:
- Matrix-vector multiplication in O(nnz) time
- Matrix transpose operation
- Memory-efficient storage (10-100x savings for sparse data)
- Comprehensive test suite
- Performance benchmarking

## How it works

The CSR format uses three arrays to represent a sparse matrix:

1. **row_offsets**: Tells you where each row starts in the other arrays
2. **column_indices**: The column position of each non-zero element
3. **nnz_values**: The actual values of the non-zero elements

For example, take this matrix:
```
[1  0  2]
[0  3  0]
[4  0  5]
```

In CSR format:
- row_offsets: [0, 2, 3, 5] (row 0 has 2 elements, row 1 has 1, row 2 has 2)
- column_indices: [0, 2, 1, 0, 2] (columns with non-zeros)
- nnz_values: [1, 2, 3, 4, 5] (the actual values)

## Building and running

You'll need a C++17 compiler. I used g++ on macOS/Linux.

```bash
# Compile and run tests
g++ -std=c++17 -O2 sparse_matrix_test.cpp SparseMatrixCSR.cpp -o tests
./tests

# Compile and run benchmarks
g++ -std=c++17 -O2 benchmark.cpp SparseMatrixCSR.cpp -o benchmark
./benchmark
```

## Usage example

```cpp
#include "SparseMatrixCSR.h"

// Create a 3x3 sparse matrix
SparseMatrixCSR matrix(3, 3);

// Build from a dense matrix
std::vector<std::vector<double>> dense = {
    {1.0, 0.0, 2.0},
    {0.0, 3.0, 0.0},
    {4.0, 0.0, 5.0}
};
matrix.buildFromDense(dense);

// Multiply by a vector
std::vector<double> vec = {1.0, 1.0, 1.0};
std::vector<double> result = matrix.multiply(vec);
// result = [3.0, 3.0, 9.0]

// Get the transpose
SparseMatrixCSR transposed = matrix.transpose();
```

## Performance results

I ran benchmarks on different matrix sizes and sparsity levels. Here are some typical results:

**2000x2000 matrix, 99% sparse:**
- Memory: 32MB (dense) vs 0.3MB (sparse) = 100x savings
- Speed: 15ms (dense) vs 0.8ms (sparse) = 18x faster
- Only 40,000 non-zero elements out of 4 million total

The speedup gets better as the matrix gets sparser:
- 90% sparse: ~5-10x faster
- 95% sparse: ~10-15x faster
- 99% sparse: ~15-20x faster

## What I learned

Building this helped me understand:
- How sparse matrix algorithms work in practice
- The trade-offs between memory usage and computation time
- Modern C++ features like const correctness and STL containers
- Performance benchmarking and optimization techniques
- The importance of choosing the right data structure for the problem

## Testing

The test suite covers:
- Basic matrix operations
- Edge cases like empty matrices
- Error handling for invalid inputs
- Transpose operations on different matrix shapes

All tests pass and verify the correctness of the implementation.

## Project structure

```
SparseMatrixLib/
├── SparseMatrixCSR.h          # Class definition
├── SparseMatrixCSR.cpp        # Implementation
├── sparse_matrix_test.cpp     # Test suite
├── benchmark.cpp              # Performance benchmarks
├── benchmark                  # Compiled benchmark
└── tests                      # Compiled tests
```

## Why I built this

I was interested in learning about sparse matrix algorithms and wanted to implement something that could actually be useful. Sparse matrices are used everywhere in scientific computing, machine learning, and graph algorithms, so understanding how to work with them efficiently is valuable.

The CSR format is one of the most common sparse matrix formats, and implementing it from scratch helped me really understand how it works under the hood.

---

This is a learning project I built to explore sparse matrix algorithms and C++ programming. Feel free to check out the code or run the benchmarks! 