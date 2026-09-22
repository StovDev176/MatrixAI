#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cctype>
#include "Matrix.hpp"

struct MNISTData {
    Matrix X; 
    Matrix Y; 
};

inline MNISTData load_mnist_csv(const std::string& filepath, size_t max_samples = 1000) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filepath << std::endl;
        return {};
    }

    std::string line;
    std::vector<float> x_data;
    std::vector<float> y_data;
    size_t count = 0;

    while (std::getline(file, line) && count < max_samples) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string val;

        if (!std::getline(ss, val, ',')) continue;

        if (!std::isdigit(val[0])) {
            continue; 
        }

        int label = std::stoi(val);

        for (int i = 0; i < 10; ++i) {
            y_data.push_back((i == label) ? 1.0f : 0.0f);
        }

        size_t pixel_count = 0;
        while (std::getline(ss, val, ',') && pixel_count < 784) {
            float pixel = std::stof(val) / 255.0f;
            x_data.push_back(pixel);
            pixel_count++;
        }

        count++;
    }

    MNISTData data;
    data.X = Matrix(count, 784);
    data.X.data = x_data;

    data.Y = Matrix(count, 10);
    data.Y.data = y_data;

    return data;
}