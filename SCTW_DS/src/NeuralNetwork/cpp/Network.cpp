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
    : layers(layers), beta1(0.9), beta2(0.999), epsilon(1e-8), iterations(0) {
  velocity_w.resize(layers.size());
  velocity_b.resize(layers.size());
  m_w.resize(layers.size());
  v_w.resize(layers.size());
  m_b.resize(layers.size());
  v_b.resize(layers.size());
  for (size_t i = 0; i < layers.size(); ++i) {
    velocity_w[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(),
                                          layers[i].weights.cols());
    velocity_b[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
    m_w[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(),
                                   layers[i].weights.cols());
    v_w[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(),
                                   layers[i].weights.cols());
    m_b[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
    v_b[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
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

std::vector<double> NeuralNetwork::classifyAll(
    const std::vector<std::vector<double>>& inputs) {
  std::vector<double> y_pred(inputs.size());
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
    const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
        training_data,
    double learning_rate, int batch_size) {
  std::vector<Eigen::MatrixXd> nabla_w_total(layers.size());
  std::vector<Eigen::VectorXd> nabla_b_total(layers.size());

  for (size_t i = 0; i < layers.size(); ++i) {
    nabla_w_total[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(),
                                             layers[i].weights.cols());
    nabla_b_total[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
  }

#pragma omp parallel for
  for (size_t k = 0; k < training_data.size(); k += batch_size) {
    std::vector<Eigen::MatrixXd> nabla_w(layers.size());
    std::vector<Eigen::VectorXd> nabla_b(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
      nabla_w[i] = Eigen::MatrixXd::Zero(layers[i].weights.rows(),
                                         layers[i].weights.cols());
      nabla_b[i] = Eigen::VectorXd::Zero(layers[i].biases.size());
    }

    for (size_t b = 0; b < batch_size && (k + b) < training_data.size(); ++b) {
      const auto& data_point = training_data[k + b];

      // Forward pass
      Eigen::VectorXd activation = data_point.first;
      std::vector<Eigen::VectorXd> activations = {activation};
      std::vector<Eigen::VectorXd> zs;

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

      for (int l = 2; l < layers.size() + 1; ++l) {
        auto sp = layers[-l + layers.size()].derivativeActivationFunction(
            zs[-l + zs.size()]);

        Eigen::VectorXd new_delta =
            (delta.transpose() *
             layers[-l + layers.size() + 1].weights.transpose())
                .cwiseProduct(sp.transpose());
        delta = new_delta;
        nabla_b[-l + nabla_b.size()] = delta;
        nabla_w[-l + nabla_w.size()] =
            activations[-l + activations.size() - 1] * delta.transpose();
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
    m_w[i] = beta1 * m_w[i] + (1 - beta1) * nabla_w_total[i];
    v_w[i] = beta2 * v_w[i] +
             (1 - beta2) * nabla_w_total[i].cwiseProduct(nabla_w_total[i]);
    Eigen::MatrixXd m_hat_w = m_w[i] / (1 - std::pow(beta1, iterations));
    Eigen::MatrixXd v_hat_w = v_w[i] / (1 - std::pow(beta2, iterations));

    m_b[i] = beta1 * m_b[i] + (1 - beta1) * nabla_b_total[i];
    v_b[i] = beta2 * v_b[i] +
             (1 - beta2) * nabla_b_total[i].cwiseProduct(nabla_b_total[i]);
    Eigen::VectorXd m_hat_b = m_b[i] / (1 - std::pow(beta1, iterations));
    Eigen::VectorXd v_hat_b = v_b[i] / (1 - std::pow(beta2, iterations));

    layers[i].weights -=
        learning_rate * m_hat_w.array()
                            .cwiseQuotient(v_hat_w.array().sqrt() + epsilon)
                            .matrix();
    layers[i].biases -=
        learning_rate * m_hat_b.array()
                            .cwiseQuotient(v_hat_b.array().sqrt() + epsilon)
                            .matrix();
  }
}

std::vector<double> NeuralNetwork::train(
    int iterations,
    const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
        data_points,
    double learning_rate, int batch_size) {
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
  const int interval = iterations / 10;
  double cost;

  for (int i = 1; i <= iterations; ++i) {
    if (stop_training) {
      break;
    }
    this->iterations = i;
    learn(eigen_data_points, learning_rate, batch_size);
    cost = this->cost(eigen_data_points);
    costs.push_back(cost);
    if (i % interval == 0) {
      p.increment(interval, cost);
    }
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
