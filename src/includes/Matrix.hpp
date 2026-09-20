#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <iostream>
struct Matrix {
    size_t columns;
    size_t rows;
    std::vector<float> data;
    Matrix(size_t r, size_t c) : columns(c), rows(r), data(r * c, 0.0f) {}

    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void random(float min_val, float max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);

        for (float& val : data) {
            val = dist(gen);
        }
    }

    float& operator()(size_t row, size_t col) {
        return data[row * columns + col];
    }

    const float& operator()(size_t row, size_t col) const {
        return data[row * columns + col];
    }

    Matrix transpose() const {
        Matrix result(columns, rows);
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < columns; c++) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    bool isSameDimension(const Matrix& matrix) const {
        return (matrix.columns == columns && matrix.rows == rows);
    }

    Matrix operator+(const Matrix& other) const {
        if (!isSameDimension(other)) {
            throw std::invalid_argument("Incompatible dimensions for the addition.");
        }
        Matrix result(rows, columns);
        size_t total_elements = rows * columns;
        for (size_t i = 0; i < total_elements; ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }

        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (this->columns != other.rows) {
            throw std::invalid_argument("Incompatible dimensions for the multiplication");
        }

        Matrix result(this->rows, other.columns);
        result.zero();

        // I'm using the IKJ multiplication for performance

        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t k = 0; k < this->columns; ++k) {
                float r = (*this)(i, k);
                for (size_t j = 0; j < other.columns; ++j) {
                    result(i, j) += r * other(k, j);
                }
            }
        }


        return result;
    }
    void print() const {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                std::cout << (*this)(r, c) << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    void apply_relu() {
        for (float& val : data) {
            if (val < 0.0f) {
                val = 0.0f;
            }
        }
    }

    void apply_sigmoid() {
        for (float& val : data) {
            val = 1.0f / (1.0f + std::exp(-val));
        }
    }
    void apply_softmax() {
        for (size_t r = 0; r < rows; ++r) {
            
            float max_val = (*this)(r, 0);
            for (size_t c = 1; c < columns; ++c) {
                if ((*this)(r, c) > max_val) {
                    max_val = (*this)(r, c);
                }
            }

            float sum_exp = 0.0f;
            for (size_t c = 0; c < columns; ++c) {
                float exp_val = std::exp((*this)(r, c) - max_val);
                (*this)(r, c) = exp_val; 
                sum_exp += exp_val;
            }

            for (size_t c = 0; c < columns; ++c) {
                (*this)(r, c) /= sum_exp;
            }
        }
    }
    void add_bias(const Matrix& bias) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                (*this)(r, c) += bias(0, c); 
            }
        }
    }   
    Matrix sum_rows() const {
        Matrix result(1, columns);
        result.zero(); 

        for (size_t c = 0; c < columns; ++c) {
            float sum = 0.0f;
            for (size_t r = 0; r < rows; ++r) {
                sum += data[r * columns + c];
            }
            result.data[c] = sum;
        }

        return result;
    } 
};

