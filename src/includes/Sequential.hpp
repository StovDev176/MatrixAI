#pragma once

#include <vector>
#include "Matrix.hpp"

struct Layer {
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& dOut) = 0;
    virtual void update(float learning_rate) {}
    virtual ~Layer() = default;
};

struct Sequential {
    std::vector<Layer*> layers;

    void add(Layer* layer) {
        layers.push_back(layer);
    }

    Matrix forward(const Matrix& input) {
        Matrix current = input;
        for (auto layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    Matrix backward(const Matrix& dLoss) {
        Matrix current_grad = dLoss;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            current_grad = layers[i]->backward(current_grad);
        }
        return current_grad;
    }

    void update(float learning_rate) {
        for (auto layer : layers) {
            layer->update(learning_rate);
        }
    }

    ~Sequential() {
        for (auto layer : layers) {
            delete layer;
        }
    }
};