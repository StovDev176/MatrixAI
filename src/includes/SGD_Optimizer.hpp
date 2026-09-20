#pragma once

#include "Dense.hpp"

struct SGDOptimizer {
    float learning_rate;

    SGDOptimizer(float lr = 0.01f) : learning_rate(lr) {}

    void update(Dense& layer) {
        for (size_t i = 0; i < layer.weights.data.size(); ++i) {
            layer.weights.data[i] -= learning_rate * layer.dW.data[i];
        }

        for (size_t i = 0; i < layer.bias.data.size(); ++i) {
            layer.bias.data[i] -= learning_rate * layer.dB.data[i];
        }
    }
};