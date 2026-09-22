#pragma once

#include "Matrix.hpp"
#include <cmath>
#include "Sequential.hpp"
#include <vector>

struct Dense : Layer {
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
    Matrix forward(const Matrix& input) override {
        this->input_cache = input;
        Matrix output = input * weights; 
        output.add_bias(bias);
        return output;
    }
    Matrix backward(const Matrix& dOut) override {
        this->dW = input_cache.transpose() * dOut;
        this->dB = dOut.sum_rows();
        Matrix dX = dOut * weights.transpose();
        return dX;
    }
    void update(float learning_rate) override {
        for (size_t i = 0; i < weights.data.size(); ++i) {
                weights.data[i] -= learning_rate * dW.data[i];
            }
        for (size_t i = 0; i < bias.data.size(); ++i) {
            bias.data[i] -= learning_rate * dB.data[i];
        }
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
