#include <iostream>
#include <vector>

#include "../header/Layers.h"

FullyConnectedLayer::FullyConnectedLayer(int in_nodes, int out_nodes,
                                         const std::string& activation_func,
                                         const std::string& cost_func)
    : num_in_nodes(in_nodes),
      num_out_nodes(out_nodes),
      activation_func(activation_func),
      cost_func(cost_func) {
  weights.resize(num_out_nodes, std::vector<double>(num_in_nodes));
  biases.resize(num_out_nodes);
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
std::pair<std::vector<double>, std::vector<double> >
FullyConnectedLayer::calculateOutputs(const std::vector<double>& inputs) {
  std::vector<double> outputs(num_out_nodes, 0.0);

  // Calculate the dot product of inputs and weights, then add biases
  for (int i = 0; i < num_out_nodes; ++i) {
    for (int j = 0; j < num_in_nodes; ++j) {
      outputs[i] += weights[i][j] * inputs[j];
    }
    outputs[i] += biases[i];
  }
  return {outputs, activationFunction(outputs)};
}

// Function Implementations
double FullyConnectedLayer::mse(const std::vector<double>& y_true,
                                const std::vector<double>& y_pred) {
  double sum = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    sum += std::pow(y_true[i] - y_pred[i], 2);
  }
  return sum / y_true.size();
}

double FullyConnectedLayer::cross_entropy(const std::vector<double>& y_true,
                                          const std::vector<double>& y_pred) {
  double sum = 0.0;
  for (size_t i = 0; i < y_true.size(); ++i) {
    sum += -y_true[i] * std::log(y_pred[i]) -
           (1 - y_true[i]) * std::log(1 - y_pred[i]);
  }
  return sum / y_true.size();
}

std::vector<double> FullyConnectedLayer::mseDerivative(
    const std::vector<double>& y_true, const std::vector<double>& y_pred) {
  std::vector<double> result(y_true.size());
  for (size_t i = 0; i < y_true.size(); ++i) {
    result[i] = 2 * (y_pred[i] - y_true[i]) / y_true.size();
  }
  return result;
}

std::vector<double> FullyConnectedLayer::cross_entropyDerivative(
    const std::vector<double>& y_true, const std::vector<double>& y_pred) {
  std::vector<double> result(y_true.size());
  for (size_t i = 0; i < y_true.size(); ++i) {
    result[i] = (y_pred[i] - y_true[i]) / (y_pred[i] * (1 - y_pred[i]));
  }
  return result;
}

std::vector<double> FullyConnectedLayer::sigmoid(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = 1 / (1 + std::exp(-x[i]));
  }
  return result;
}

std::vector<double> FullyConnectedLayer::relu(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::max(0.0, x[i]);
  }
  return result;
}

std::vector<double> FullyConnectedLayer::leaky_relu(
    const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = (x[i] > 0) ? x[i] : 0.01 * x[i];
  }
  return result;
}

std::vector<double> FullyConnectedLayer::tanh(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::tanh(x[i]);
  }
  return result;
}

std::vector<double> FullyConnectedLayer::softmax(const std::vector<double>& x) {
  std::vector<double> result(x.size());
  double max = *std::max_element(x.begin(), x.end());
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::exp(x[i] - max);
    sum += result[i];
  }
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] /= sum;
  }
  return result;
}

std::vector<double> FullyConnectedLayer::sigmoid_derivative(
    const std::vector<double>& x) {
  std::vector<double> sigmoid_vals = sigmoid(x);
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = sigmoid_vals[i] * (1 - sigmoid_vals[i]);
  }
  return result;
}

std::vector<double> FullyConnectedLayer::relu_derivative(
    const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = (x[i] > 0) ? 1.0 : 0.0;
  }
  return result;
}

std::vector<double> FullyConnectedLayer::leaky_relu_derivative(
    const std::vector<double>& x) {
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = (x[i] > 0) ? 1.0 : 0.01;
  }
  return result;
}

std::vector<double> FullyConnectedLayer::tanh_derivative(
    const std::vector<double>& x) {
  std::vector<double> tanh_vals = tanh(x);
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = 1 - std::pow(tanh_vals[i], 2);
  }
  return result;
}

std::vector<double> FullyConnectedLayer::softmax_derivative(
    const std::vector<double>& x) {
  // This is a simplified version of the derivative for illustrative purposes
  std::vector<double> softmax_vals = softmax(x);
  std::vector<double> result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = softmax_vals[i] * (1 - softmax_vals[i]);
  }
  return result;
}
