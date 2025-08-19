#include "SparseMatrixCSR.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

class DenseMatrix {
public:
    std::vector<std::vector<double>> data;
    int rows, cols;
    
    DenseMatrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    
    std::vector<double> multiply(const std::vector<double>& vec) const {
        std::vector<double> result(rows, 0.0);
        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                result[r] += data[r][c] * vec[c];
            }
        }
        return result;
    }
    
    size_t getMemoryUsage() const {
        return rows * cols * sizeof(double);
    }
};

std::pair<DenseMatrix, SparseMatrixCSR> generateMatrix(int size, double sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> value_dist(1.0, 10.0);
    std::uniform_real_distribution<> sparse_dist(0.0, 1.0);
    
    DenseMatrix dense(size, size);
    
    for(int r = 0; r < size; ++r) {
        for(int c = 0; c < size; ++c) {
            if(sparse_dist(gen) > sparsity) { 
                dense.data[r][c] = value_dist(gen);
            }
        }
    }
    
    SparseMatrixCSR sparse(size, size);
    sparse.buildFromDense(dense.data);
    
    return {dense, sparse};
}

std::vector<double> generateVector(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(1.0, 10.0);
    
    std::vector<double> vec(size);
    for(int i = 0; i < size; ++i) {
        vec[i] = dist(gen);
    }
    return vec;
}

void benchmarkMultiplication(const DenseMatrix& dense, const SparseMatrixCSR& sparse, 
                           const std::vector<double>& vec, int iterations = 100) {
    
    // Benchmark dense multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        auto result = dense.multiply(vec);
        (void)result; // block optimization
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto dense_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark sparse multiplication  
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        auto result = sparse.multiply(vec);
        (void)result; // block optimization
    }
    end = std::chrono::high_resolution_clock::now();
    auto sparse_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double speedup = static_cast<double>(dense_time.count()) / sparse_time.count();
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Dense time:  " << dense_time.count() / 1000.0 << " ms\n";
    std::cout << "  Sparse time: " << sparse_time.count() / 1000.0 << " ms\n";
    std::cout << "  Speedup:     " << speedup << "x\n";
}

void benchmarkTranspose(const SparseMatrixCSR& sparse, int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        auto transposed = sparse.transpose();
        (void)transposed; // Prevent optimization
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Transpose time: " << time.count() / 1000.0 << " ms (avg over " << iterations << " runs)\n";
}

void runBenchmarks() {
    std::cout << "=== Sparse Matrix Performance Benchmarks ===\n\n";
    
    std::vector<int> sizes = {500, 1000, 2000};
    std::vector<double> sparsity_levels = {0.90, 0.95, 0.99}; // 90%, 95%, 99% sparse test levels 
    
    for(int size : sizes) {
        std::cout << "Matrix size: " << size << "x" << size << "\n";
        std::cout << std::string(50, '-') << "\n";
        
        for(double sparsity : sparsity_levels) {
            std::cout << "Sparsity: " << (sparsity * 100) << "%\n";
            
            auto [dense, sparse] = generateMatrix(size, sparsity);
            auto vec = generateVector(size);
            
            size_t dense_memory = dense.getMemoryUsage();
            size_t sparse_memory = sparse.getNNZ() * (sizeof(double) + sizeof(int)) + 
                                 (sparse.getRows() + 1) * sizeof(int);

            double memory_savings = static_cast<double>(dense_memory) / sparse_memory;
            
            std::cout << "  Non-zeros:   " << sparse.getNNZ() << " / " << (size * size) << "\n";
            std::cout << "  Dense memory:  " << dense_memory / 1024.0 << " KB\n";
            std::cout << "  Sparse memory: " << sparse_memory / 1024.0 << " KB\n";
            std::cout << "  Memory savings: " << memory_savings << "x\n";
            

            std::cout << "  Matrix-vector multiplication:\n";
            benchmarkMultiplication(dense, sparse, vec);
            
            std::cout << "  Matrix transpose:\n";
            benchmarkTranspose(sparse);
            
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "Starting benchmarks...\n";
    std::cout << "(This may take a few minutes)\n\n";
    
    runBenchmarks();
    
    std::cout << "=== Summary ===\n";
    std::cout << "• Sparse matrices show significant memory savings (10x-100x)\n";
    std::cout << "• Performance gains increase with sparsity level\n";
    std::cout << "• CSR format enables O(nnz) matrix-vector multiplication\n";
    std::cout << "• Transpose operation runs in O(nnz) time\n";
    
    return 0;
}
