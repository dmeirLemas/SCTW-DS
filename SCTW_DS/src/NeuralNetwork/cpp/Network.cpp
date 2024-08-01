#include "../header/Network.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../misc/ProgressBar.cpp"

// Signal handler for keyboard interrupt
volatile sig_atomic_t stop_training = 0;

void handle_signal(int signal) { stop_training = 1; }

using Eigen::MatrixXd;
using Eigen::VectorXd;

NeuralNetwork::NeuralNetwork(const std::vector<FullyConnectedLayer>& layers)
    : layers(layers) {
  velocity_w.resize(layers.size());
  velocity_b.resize(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    velocity_w[i] =
        MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols());
    velocity_b[i] = VectorXd::Zero(layers[i].biases.size());
  }
}

VectorXd NeuralNetwork::calculateOutputs(const VectorXd& inputs) {
  VectorXd activations = inputs;
  for (auto& layer : layers) {
    activations = layer.calculateOutputs(activations).second;
  }
  return activations;
}

double NeuralNetwork::classify(const VectorXd& inputs) {
  VectorXd outputs = calculateOutputs(inputs);
  return static_cast<double>(std::distance(
      outputs.data(),
      std::max_element(outputs.data(), outputs.data() + outputs.size())));
}

VectorXd NeuralNetwork::classifyAll(
    const std::vector<std::vector<double>>& inputs) {
  VectorXd y_pred(inputs.size());
#pragma omp parallel for
  for (int i = 0; i < inputs.size(); ++i) {
    VectorXd eigen_input = VectorXd::Map(inputs[i].data(), inputs[i].size());
    y_pred[i] = classify(eigen_input);
  }
  return y_pred;
}

double NeuralNetwork::cost(
    const std::vector<std::pair<VectorXd, VectorXd>>& data_points) {
  double total_cost = 0.0;
#pragma omp parallel for reduction(+ : total_cost)
  for (size_t i = 0; i < data_points.size(); ++i) {
    const auto& data_point = data_points[i];
    VectorXd y_pred = calculateOutputs(data_point.first);
    const VectorXd& y_true = data_point.second;
    total_cost += layers.back().costFunction(y_true, y_pred);
  }
  return total_cost / data_points.size();
}

void NeuralNetwork::learn(
    const std::vector<std::pair<VectorXd, VectorXd>>& training_data,
    double learning_rate, int batch_size, double momentum) {
  std::vector<MatrixXd> nabla_w_total(layers.size());
  std::vector<VectorXd> nabla_b_total(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    nabla_w_total[i] =
        MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols());
    nabla_b_total[i] = VectorXd::Zero(layers[i].biases.size());
  }

#pragma omp parallel for
  for (size_t k = 0; k < training_data.size(); k += batch_size) {
    std::vector<MatrixXd> nabla_w(layers.size());
    std::vector<VectorXd> nabla_b(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
      nabla_w[i] =
          MatrixXd::Zero(layers[i].weights.rows(), layers[i].weights.cols());
      nabla_b[i] = VectorXd::Zero(layers[i].biases.size());
    }

    for (size_t b = 0; b < batch_size && (k + b) < training_data.size(); ++b) {
      const auto& data_point = training_data[k + b];

      // Forward pass
      VectorXd activation = data_point.first;
      std::vector<VectorXd> activations = {activation};
      std::vector<VectorXd> zs;

      for (auto& layer : layers) {
        auto res = layer.calculateOutputs(activation);
        auto z = res.first;
        activation = res.second;
        zs.push_back(z);
        activations.push_back(activation);
      }

      // Backward pass
      auto delta = layers.back().derivativeCostFunction(data_point.second,
                                                        activations.back());
      for (size_t i = 0; i < delta.size(); ++i) {
        delta[i] *= layers.back().derivativeActivationFunction(zs.back())[i];
      }

      nabla_b.back() = delta;
      nabla_w.back() = activations[activations.size() - 2] * delta.transpose();

      for (int l = layers.size() - 2; l >= 0; --l) {
        auto sp = layers[l].derivativeActivationFunction(zs[l]);
        VectorXd new_delta =
            (layers[l + 1].weights.transpose() * delta).cwiseProduct(sp);
        delta = new_delta;
        nabla_b[l] = delta;
        nabla_w[l] = activations[l] * delta.transpose();
      }
    }

#pragma omp critical
    {
      for (size_t i = 0; i < layers.size(); ++i) {
        nabla_b_total[i] += nabla_b[i];
        nabla_w_total[i] += nabla_w[i];
      }
    }
  }

  for (size_t i = 0; i < layers.size(); ++i) {
    velocity_b[i] = momentum * velocity_b[i] -
                    (learning_rate / batch_size) * nabla_b_total[i];
    layers[i].biases += velocity_b[i];
    velocity_w[i] = momentum * velocity_w[i] -
                    (learning_rate / batch_size) * nabla_w_total[i];
    layers[i].weights += velocity_w[i];
  }
}

std::vector<double> NeuralNetwork::train(
    int iterations,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        data_points,
    double learning_rate, int batch_size, double momentum) {
  // Convert std::vector<std::pair<std::vector<double>, std::vector<double>>> to
  // std::vector<std::pair<VectorXd, VectorXd>>
  std::vector<std::pair<VectorXd, VectorXd>> eigen_data_points;
  for (const auto& dp : data_points) {
    VectorXd input = VectorXd::Map(dp.first.data(), dp.first.size());
    VectorXd output = VectorXd::Map(dp.second.data(), dp.second.size());
    eigen_data_points.emplace_back(input, output);
  }

  std::vector<double> costs;
  std::signal(SIGINT, handle_signal);
  ProgressBar p = ProgressBar(iterations);

  for (int i = 0; i < iterations; ++i) {
    if (stop_training) {
      break;
    }
    learn(eigen_data_points, learning_rate, batch_size, momentum);
    double cost = this->cost(eigen_data_points);
    costs.push_back(cost);
    p.increment(1, cost);
  }
  return costs;
}

void NeuralNetwork::saveModel(const std::string& file_path) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file to save model.");
  }
  for (const auto& layer : layers) {
    file.write(reinterpret_cast<const char*>(layer.weights.data()),
               layer.weights.size() * sizeof(double));
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
    file.read(reinterpret_cast<char*>(layer.weights.data()),
              layer.weights.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(layer.biases.data()),
              layer.biases.size() * sizeof(double));
    layers.push_back(layer);
  }
  return NeuralNetwork(layers);
}
