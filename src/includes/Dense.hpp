#pragma once

#include "Matrix.hpp"
#include <cmath>
#include <vector>

struct Dense {
    Matrix weights;
    Matrix bias;  
    Matrix input_cache; 
    Matrix dW; // mesure l'impact des poids sur le résultat
    Matrix dB; // mesure l'impact du biais sur le résultat

    Dense(size_t input_dim, size_t output_dim) 
        : weights(input_dim, output_dim), bias(1, output_dim) {
        
        weights.random(-0.5f, 0.5f);
        bias.zero();
    }

    Matrix forward(const Matrix& input) {
        this->input_cache = input;
        Matrix output = input * weights; 
        output.add_bias(bias);
        return output;
    }
    void relu_backward(Matrix& dOut, const Matrix& x) {
        for (size_t i = 0; i < dOut.data.size(); ++i) {
            if (x.data[i] <= 0.0f) {
                dOut.data[i] = 0.0f; 
            }
        }
    }
    void sigmoid_backward(Matrix& dOut, const Matrix& a) {
        for (size_t i = 0; i < dOut.data.size(); ++i) {
            float sig = a.data[i];
            float derivative = sig * (1.0f - sig);
            dOut.data[i] *= derivative; 
        }
    }
    Matrix backward(const Matrix& dOut) {
        this->dW = input_cache.transpose() * dOut;
        this->dB = dOut.sum_rows();
        Matrix dX = dOut * weights.transpose();
        return dX;
    }
};

struct MSELoss {
    float forward(const Matrix& y_pred, const Matrix& y_true) {
        float total_loss = 0.0f;
        size_t samples = y_pred.rows;

        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            float diff = y_pred.data[i] - y_true.data[i];
            total_loss += diff * diff;
        }
        
        return total_loss / (2.0f * samples); 
    }

    Matrix backward(const Matrix& y_pred, const Matrix& y_true) {
        Matrix dOut(y_pred.rows, y_pred.columns);
        size_t samples = y_pred.rows;

        for (size_t i = 0; i < y_pred.data.size(); ++i) {
            dOut.data[i] = (y_pred.data[i] - y_true.data[i]) / static_cast<float>(samples);
        }

        return dOut;
    }
};
