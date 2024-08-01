#include "../header/FullyConnectedLayer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

FullyConnectedLayer::FullyConnectedLayer(int in_nodes, int out_nodes,
                                         const std::string& activation_func,
                                         const std::string& cost_func)
    : num_in_nodes(in_nodes),
      num_out_nodes(out_nodes),
      activation_func(activation_func),
      cost_func(cost_func) {
  weights = Eigen::MatrixXd::Random(num_in_nodes, num_out_nodes);
  biases = Eigen::VectorXd::Random(num_out_nodes);

  cost_map["mse"] = std::bind(&FullyConnectedLayer::mse, this,
                              std::placeholders::_1, std::placeholders::_2);
  cost_map["cross_entropy"] =
      std::bind(&FullyConnectedLayer::cross_entropy, this,
                std::placeholders::_1, std::placeholders::_2);

  // Initialize activation_map
  activation_map["sigmoid"] =
      std::bind(&FullyConnectedLayer::sigmoid, this, std::placeholders::_1);
  activation_map["relu"] =
      std::bind(&FullyConnectedLayer::relu, this, std::placeholders::_1);
  activation_map["leaky_relu"] =
      std::bind(&FullyConnectedLayer::leaky_relu, this, std::placeholders::_1);
  activation_map["tanh"] =
      std::bind(&FullyConnectedLayer::tanh, this, std::placeholders::_1);
  activation_map["softmax"] =
      std::bind(&FullyConnectedLayer::softmax, this, std::placeholders::_1);

  // Initialize cost_derivative_map
  cost_derivative_map["mse"] =
      std::bind(&FullyConnectedLayer::mseDerivative, this,
                std::placeholders::_1, std::placeholders::_2);
  cost_derivative_map["cross_entropy"] =
      std::bind(&FullyConnectedLayer::cross_entropyDerivative, this,
                std::placeholders::_1, std::placeholders::_2);

  // Initialize derivative_map
  activation_derivative_map["sigmoid"] = std::bind(
      &FullyConnectedLayer::sigmoid_derivative, this, std::placeholders::_1);
  activation_derivative_map["relu"] = std::bind(
      &FullyConnectedLayer::relu_derivative, this, std::placeholders::_1);
  activation_derivative_map["leaky_relu"] = std::bind(
      &FullyConnectedLayer::leaky_relu_derivative, this, std::placeholders::_1);
  activation_derivative_map["tanh"] = std::bind(
      &FullyConnectedLayer::tanh_derivative, this, std::placeholders::_1);
  activation_derivative_map["softmax"] = std::bind(
      &FullyConnectedLayer::softmax_derivative, this, std::placeholders::_1);

  // Set Functions
  setFunctions(cost_func, activation_func);
}

void FullyConnectedLayer::setFunctions(const std::string& cost_func,
                                       const std::string& activation_func) {
  if (cost_map.find(cost_func) != cost_map.end()) {
    costFunction = cost_map[cost_func];
    derivativeCostFunction = cost_derivative_map[cost_func];
  } else {
    throw std::invalid_argument("Cost function not found: " + cost_func);
  }

  if (activation_map.find(activation_func) != activation_map.end()) {
    activationFunction = activation_map[activation_func];
    derivativeActivationFunction = activation_derivative_map[activation_func];
  } else {
    throw std::invalid_argument("Activation function not found: " +
                                activation_func);
  }
}

// Calculate outputs method
std::pair<Eigen::VectorXd, Eigen::VectorXd>
FullyConnectedLayer::calculateOutputs(const Eigen::VectorXd& inputs) {
  Eigen::VectorXd z = inputs * weights.transpose() + biases;
  Eigen::VectorXd activation = activationFunction(z);
  return {z, activation};
}

// Function Implementations
double FullyConnectedLayer::mse(const Eigen::VectorXd& y_true,
                                const Eigen::VectorXd& y_pred) {
  return (y_true - y_pred).squaredNorm() / y_true.size();
}

double FullyConnectedLayer::cross_entropy(const Eigen::VectorXd& y_true,
                                          const Eigen::VectorXd& y_pred) {
  return -(y_true.array() * y_pred.array().log() +
           (1 - y_true.array()) * (1 - y_pred.array()).log())
              .mean();
}

Eigen::VectorXd FullyConnectedLayer::mseDerivative(
    const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
  return 2 * (y_pred - y_true) / y_true.size();
}

Eigen::VectorXd FullyConnectedLayer::cross_entropyDerivative(
    const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
  return (y_pred - y_true).array() / (y_pred.array() * (1 - y_pred.array()));
}

Eigen::VectorXd FullyConnectedLayer::sigmoid(const Eigen::VectorXd& x) {
  return 1 / (1 + (-x.array()).exp());
}

Eigen::VectorXd FullyConnectedLayer::relu(const Eigen::VectorXd& x) {
  return x.array().max(0);
}

Eigen::VectorXd FullyConnectedLayer::leaky_relu(const Eigen::VectorXd& x) {
  return x.array().max(0) + 0.01 * x.array().min(0);
}

Eigen::VectorXd FullyConnectedLayer::tanh(const Eigen::VectorXd& x) {
  return x.array().tanh();
}

Eigen::VectorXd FullyConnectedLayer::softmax(const Eigen::VectorXd& x) {
  Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
  return exp_x / exp_x.sum();
}

Eigen::VectorXd FullyConnectedLayer::sigmoid_derivative(
    const Eigen::VectorXd& x) {
  Eigen::VectorXd sig = sigmoid(x);
  return sig.array() * (1 - sig.array());
}

Eigen::VectorXd FullyConnectedLayer::relu_derivative(const Eigen::VectorXd& x) {
  return (x.array() > 0).cast<double>();
}

Eigen::VectorXd FullyConnectedLayer::leaky_relu_derivative(
    const Eigen::VectorXd& x) {
  return (x.array() > 0).cast<double>() +
         0.01 * (x.array() <= 0).cast<double>();
}

Eigen::VectorXd FullyConnectedLayer::tanh_derivative(const Eigen::VectorXd& x) {
  return 1 - tanh(x).array().square();
}

Eigen::VectorXd FullyConnectedLayer::softmax_derivative(
    const Eigen::VectorXd& x) {
  Eigen::VectorXd soft = softmax(x);
  return soft.array() * (1 - soft.array());
}
