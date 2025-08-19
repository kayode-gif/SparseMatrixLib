
#include "SparseMatrixCSR.h"


void SparseMatrixCSR::buildFromDense(const std::vector<std::vector<double>>& dense){
    if(dense.size() != rows){
        throw std::invalid_argument("Dense matrix row count doesn't match");
    }

    for(int r = 0; r < rows; ++r){
        if(dense[r].size() != cols){
            throw std::invalid_argument("Dense matrix has inconsistent column sizes");
        }
    }

    nnz = 0;  

    column_indices.clear();
    nnz_values.clear();

    row_offsets[0] = 0;

    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            double value = dense[r][c];
            if(value != 0){
                column_indices.push_back(c);
                nnz_values.push_back(value);
                ++nnz;
            }
        }
        row_offsets[r+1] = nnz;
    }
}

std::vector<double> SparseMatrixCSR::multiply(const std::vector<double>& dense_vector) const{
    if(dense_vector.size() != cols){
        throw std::invalid_argument("Vector size does not match matrix colums");
    }
    std::vector<double> result(rows, 0.0);
    for(int row = 0; row < rows; ++row){
        double sum = 0.0;
        for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx){
            int col = column_indices[idx];
            double value = nnz_values[idx];
            sum += value * dense_vector[col];
        }
        result[row] = sum;
    }
    return result;
}

SparseMatrixCSR SparseMatrixCSR::transpose() const{
    // count how many elements in each col
    std::vector<int> col_ele_counts(cols, 0);
    for(int col: column_indices){
        col_ele_counts[col]++;
    }
    std::vector<int> new_row_offsets(cols + 1, 0);
    for(int i = 0; i < cols; ++i){
        new_row_offsets[i + 1] = new_row_offsets[i] + col_ele_counts[i];
    }

    SparseMatrixCSR transposed(cols, rows);  // swap dimensions
    transposed.row_offsets = new_row_offsets;
    transposed.column_indices.resize(nnz);
    transposed.nnz_values.resize(nnz);
    transposed.nnz = nnz;

    std::vector<int> col_counters = col_ele_counts; 

    for(int row = 0; row < rows; ++row) {
        for(int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
            int col = column_indices[idx];
            double value = nnz_values[idx];
            
            // in transposed matrix the old_col becomes new_row, old_row becomes new_col
            int new_row = col;
            int new_col = row;
            
            // look where to put this element in the new matrix
            int new_position = new_row_offsets[new_row] + (col_ele_counts[new_row] - col_counters[new_row]);
            
            transposed.column_indices[new_position] = new_col;
            transposed.nnz_values[new_position] = value;
            
            col_counters[new_row]--; 
        }
    }

    return transposed;

}

int SparseMatrixCSR::getRows() const{
    return rows;
}

int SparseMatrixCSR::getCols() const{
    return cols;
}

int SparseMatrixCSR::getNNZ() const{
    return nnz;
}
