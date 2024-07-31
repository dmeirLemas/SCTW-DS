#include "../header/Network.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

NeuralNetwork::NeuralNetwork(const std::vector<FullyConnectedLayer>& layers)
    : layers(layers) {
  velocity_w.resize(layers.size());
  velocity_b.resize(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    velocity_w[i].resize(layers[i].weights.size(),
                         std::vector<double>(layers[i].weights[0].size(), 0.0));
    velocity_b[i].resize(layers[i].biases.size(), 0.0);
  }
}

std::vector<double> NeuralNetwork::calculateOutputs(
    const std::vector<double>& inputs) {
  std::vector<double> activations = inputs;
  for (auto& layer : layers) {
    activations = layer.calculateOutputs(activations).second;
  }
  return activations;
}

int NeuralNetwork::classify(const std::vector<double>& inputs) {
  std::vector<double> outputs = calculateOutputs(inputs);
  return std::distance(outputs.begin(),
                       std::max_element(outputs.begin(), outputs.end()));
}

double NeuralNetwork::cost(
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        data_points) {
  double total_cost = 0.0;
  for (const auto& data_point : data_points) {
    std::vector<double> y_pred = calculateOutputs(data_point.first);
    const std::vector<double>& y_true = data_point.second;
    total_cost += layers.back().costFunction(y_true, y_pred);
  }
  return total_cost / data_points.size();
}

void NeuralNetwork::learn(
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        training_data,
    double learning_rate, int batch_size, double momentum) {
  std::vector<std::vector<std::vector<double>>> nabla_w_total(layers.size());
  std::vector<std::vector<double>> nabla_b_total(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    nabla_w_total[i].resize(
        layers[i].weights.size(),
        std::vector<double>(layers[i].weights[0].size(), 0.0));
    nabla_b_total[i].resize(layers[i].biases.size(), 0.0);
  }

  for (size_t k = 0; k < training_data.size(); k += batch_size) {
    std::vector<std::vector<std::vector<double>>> nabla_w(layers.size());
    std::vector<std::vector<double>> nabla_b(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
      nabla_w[i].resize(layers[i].weights.size(),
                        std::vector<double>(layers[i].weights[0].size(), 0.0));
      nabla_b[i].resize(layers[i].biases.size(), 0.0);
    }

    for (size_t b = 0; b < batch_size && (k + b) < training_data.size(); ++b) {
      const auto& data_point = training_data[k + b];

      // Forward pass
      std::vector<double> activation = data_point.first;
      std::vector<std::vector<double>> activations = {activation};
      std::vector<std::vector<double>> zs;

      for (auto& layer : layers) {
        std::pair<std::vector<double>, std::vector<double>> res =
            layer.calculateOutputs(activation);
        std::vector<double> z = res.first;
        std::vector<double> activation = res.second;
        zs.push_back(z);
        activations.push_back(activation);
      }

      // Backward pass
      std::vector<double> delta =
          layers.back().derivativeActivationFunction(zs.back());
      for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] *= layers.back().derivativeCostFunction(activations.back(),
                                                         data_point.second)[i];
      }

      nabla_b.back() = delta;
      for (size_t i = 0; i < delta.size(); ++i) {
        for (size_t j = 0; j < activations[activations.size() - 2].size();
             ++j) {
          nabla_w.back()[i][j] =
              activations[activations.size() - 2][j] * delta[i];
        }
      }

      for (int l = layers.size() - 2; l >= 0; --l) {
        std::vector<double> sp = layers[l].derivativeActivationFunction(zs[l]);
        std::vector<double> new_delta(layers[l].biases.size(), 0.0);
        for (size_t i = 0; i < new_delta.size(); ++i) {
          for (size_t j = 0; j < delta.size(); ++j) {
            new_delta[i] += delta[j] * layers[l + 1].weights[j][i];
          }
          new_delta[i] *= sp[i];
        }
        delta = new_delta;
        nabla_b[l] = delta;
        for (size_t i = 0; i < delta.size(); ++i) {
          for (size_t j = 0; j < activations[l].size(); ++j) {
            nabla_w[l][i][j] = activations[l][j] * delta[i];
          }
        }
      }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
      for (size_t j = 0; j < nabla_b[i].size(); ++j) {
        nabla_b_total[i][j] += nabla_b[i][j];
      }
      for (size_t j = 0; j < nabla_w[i].size(); ++j) {
        for (size_t k = 0; k < nabla_w[i][j].size(); ++k) {
          nabla_w_total[i][j][k] += nabla_w[i][j][k];
        }
      }
    }
  }

  for (size_t i = 0; i < layers.size(); ++i) {
    for (size_t j = 0; j < layers[i].biases.size(); ++j) {
      velocity_b[i][j] = momentum * velocity_b[i][j] -
                         (learning_rate / batch_size) * nabla_b_total[i][j];
      layers[i].biases[j] += velocity_b[i][j];
    }
    for (size_t j = 0; j < layers[i].weights.size(); ++j) {
      for (size_t k = 0; k < layers[i].weights[j].size(); ++k) {
        velocity_w[i][j][k] =
            momentum * velocity_w[i][j][k] -
            (learning_rate / batch_size) * nabla_w_total[i][j][k];
        layers[i].weights[j][k] += velocity_w[i][j][k];
      }
    }
  }
}

std::vector<double> NeuralNetwork::train(
    int iterations,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        data_points,
    double learning_rate, int batch_size, double momentum) {
  std::vector<double> costs;
  for (int i = 0; i < iterations; ++i) {
    learn(data_points, learning_rate, batch_size, momentum);
    double cost = this->cost(data_points);
    costs.push_back(cost);
    std::cout << "Iteration " << i + 1 << "/" << iterations << " cost: " << cost
              << std::endl;
  }
  return costs;
}

void NeuralNetwork::saveModel(const std::string& file_path) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file to save model.");
  }
  for (const auto& layer : layers) {
    for (const auto& row : layer.weights) {
      file.write(reinterpret_cast<const char*>(row.data()),
                 row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char*>(layer.biases.data()),
               layer.biases.size() * sizeof(double));
  }
}

NeuralNetwork NeuralNetwork::loadModel(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file to load model.");
  }

  std::vector<FullyConnectedLayer> layers;
  while (file.peek() != EOF) {
    FullyConnectedLayer layer(0, 0, "", "");  // Dummy initialization
    for (auto& row : layer.weights) {
      file.read(reinterpret_cast<char*>(row.data()),
                row.size() * sizeof(double));
    }
    file.read(reinterpret_cast<char*>(layer.biases.data()),
              layer.biases.size() * sizeof(double));
    layers.push_back(layer);
  }
  return NeuralNetwork(layers);
}
