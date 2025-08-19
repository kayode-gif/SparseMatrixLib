#ifndef SparseMatrixCSR_H
#define SparseMatrixCSR_H

#include <vector>


class SparseMatrixCSR{

private:
    int rows; // number of rows in the matrix 
    int cols; // number of cols in the matrix 
    int nnz; // the number of non-zero elements in the matrix 

    std::vector<int> row_offsets; // row start indices (size = nrows + 1)
    std::vector<int> column_indices; // pointers to the column indicies of non zeros (size = nnz)
    std::vector<double> nnz_values; // size = nnz, holds non-zero values in row-major ordering



public:
    SparseMatrixCSR(int r, int c): rows(r), cols(c), nnz(0), row_offsets(r + 1, 0) {};

    void buildFromDense(const std::vector<std::vector<double>>& dense);
    std::vector<double> multiply(const std::vector<double>& dense_vector) const;
    SparseMatrixCSR transpose() const;


    int getRows() const;
    int getCols() const;
    int getNNZ() const;

};

#endif // SparseMatrixCSR_H


