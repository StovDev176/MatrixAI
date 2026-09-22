#include <iostream>
#include <algorithm>
#include "includes/Matrix.hpp"
#include "includes/Sequential.hpp"
#include "includes/Dense.hpp"
#include "includes/Math.hpp"
#include "includes/MNISTLoader.hpp"

float compute_accuracy(const Matrix& predictions, const Matrix& targets) {
    int correct = 0;
    for (size_t r = 0; r < predictions.rows; ++r) {
        int pred_label = 0;
        float max_pred = predictions(r, 0);
        for (size_t c = 1; c < predictions.columns; ++c) {
            if (predictions(r, c) > max_pred) {
                max_pred = predictions(r, c);
                pred_label = static_cast<int>(c);
            }
        }

        int true_label = 0;
        for (size_t c = 0; c < targets.columns; ++c) {
            if (targets(r, c) == 1.0f) {
                true_label = static_cast<int>(c);
                break;
            }
        }

        if (pred_label == true_label) {
            correct++;
        }
    }
    return (static_cast<float>(correct) / predictions.rows) * 100.0f;
}

int main() {
    std::cout << "Beginning of the loading...\n";
    MNISTData train_data = load_mnist_csv("includes/mnist_test.csv", 8000);

    if (train_data.X.rows == 0) {
        return 1;
    }

    std::cout << "Loaded data : " << train_data.X.rows << " images of size 28*28(784 pixels).\n\n";

    Sequential model;
    model.add(new Dense(784, 128));
    model.add(new ReLU());
    model.add(new Dense(128, 10));
    model.add(new Sigmoid()); 

    MSELoss loss_fn;
    float learning_rate = 2.0f;
    int epochs = 50;
    size_t batch_size = 16;

    std::cout << "Start of the training on MNIST images\n";
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_epoch_loss = 0.0f;
        int num_batches = 0;

        for (size_t i = 0; i < train_data.X.rows; i += batch_size) {
            Matrix X_batch = train_data.X.get_row_range(i, batch_size);
            Matrix Y_batch = train_data.Y.get_row_range(i, batch_size);

            Matrix y_pred = model.forward(X_batch);
            float loss = loss_fn.forward(y_pred, Y_batch);
            Matrix dLoss = loss_fn.backward(y_pred, Y_batch);
            
            model.backward(dLoss);
            model.update(learning_rate);

            total_epoch_loss += loss;
            num_batches++;
        }
        std::cout << epoch << "\n";

        if (epoch % 5 == 0 || epoch == 1) {
            Matrix full_pred = model.forward(train_data.X);
            float acc = compute_accuracy(full_pred, train_data.Y);
            std::cout << "Epoch " << epoch << "/" << epochs 
                   << " | Loss: " << total_epoch_loss / num_batches 
                    << " | Precision: " << acc << "%\n";
        }
    }

    std::cout << "\n Training finished \n";
    return 0;
}