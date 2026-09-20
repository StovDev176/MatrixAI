#pragma once

#include <iostream>
#include <cmath>
#include <vector>

#define e 2.71828

inline void ReLU(float& x) {
    if (x < 0) x = 0.0f;
}

inline void sigmoid(float& x) {
    x = 1.0f/(1.0f+std::exp(-x));
}

void softmax(std::vector<float>& scores) {
    for (int i=0; i<scores.size(); ++i) {
        float exp = std::exp(scores[i]);
        float sum = 0.0f;
        for ()
    }
}
