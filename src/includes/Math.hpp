#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include "Sequential.hpp"

struct ReLU : Layer {
    Matrix input_cache;

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        Matrix output(input.rows, input.columns);

        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = (input.data[i] > 0.0f) ? input.data[i] : 0.0f;
        }
        return output;
    }

    Matrix backward(const Matrix& dOut) override {
        Matrix dX(dOut.rows, dOut.columns);

        for (size_t i = 0; i < dOut.data.size(); ++i) {
            dX.data[i] = (input_cache.data[i] > 0.0f) ? dOut.data[i] : 0.0f;
        }
        return dX;
    }
};


struct Sigmoid : Layer {
    Matrix output_cache; 

    Matrix forward(const Matrix& input) override {
        Matrix output(input.rows, input.columns);

        for (size_t i = 0; i < input.data.size(); ++i) {
            output.data[i] = 1.0f / (1.0f + std::exp(-input.data[i]));
        }

        this->output_cache = output;
        return output;
    }

    Matrix backward(const Matrix& dOut) override {
        Matrix dX(dOut.rows, dOut.columns);

        for (size_t i = 0; i < dOut.data.size(); ++i) {
            float y = this->output_cache.data[i];
            dX.data[i] = dOut.data[i] * y * (1.0f - y);
        }
        return dX;
    }
};