#ifndef LAYERS_H
#define LAYERS_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

class FullyConnectedLayer {
 private:
  // Shapes
  int num_in_nodes;
  int num_out_nodes;

  // Activation And Cost Function Names
  std::string activation_func;
  std::string cost_func;

  // Cost Functions
  double mse(const std::vector<double>& y_true,
             const std::vector<double>& y_pred);
  double cross_entropy(const std::vector<double>& y_true,
                       const std::vector<double>& y_pred);

  // Cost Function Derivatives
  std::vector<double> mseDerivative(const std::vector<double>& y_true,
                                    const std::vector<double>& y_pred);
  std::vector<double> cross_entropyDerivative(
      const std::vector<double>& y_true, const std::vector<double>& y_pred);

  // Activation Functions
  std::vector<double> sigmoid(const std::vector<double>& x);
  std::vector<double> relu(const std::vector<double>& x);
  std::vector<double> leaky_relu(const std::vector<double>& x);
  std::vector<double> tanh(const std::vector<double>& x);
  std::vector<double> softmax(const std::vector<double>& x);

  // Activation Function Derivatives
  std::vector<double> sigmoid_derivative(const std::vector<double>& x);
  std::vector<double> relu_derivative(const std::vector<double>& x);
  std::vector<double> leaky_relu_derivative(const std::vector<double>& x);
  std::vector<double> tanh_derivative(const std::vector<double>& x);
  std::vector<double> softmax_derivative(const std::vector<double>& x);

  // Necessary Internal Function Definitions For Ease
  using CostFunction = std::function<double(const std::vector<double>&,
                                            const std::vector<double>&)>;
  using ActivationFunction =
      std::function<std::vector<double>(const std::vector<double>&)>;
  using DerivativeCostFunction = std::function<std::vector<double>(
      const std::vector<double>&, const std::vector<double>&)>;
  using DerivativeActivationFunction =
      std::function<std::vector<double>(const std::vector<double>&)>;

  // Maps
  std::unordered_map<std::string, CostFunction> cost_map;
  std::unordered_map<std::string, ActivationFunction> activation_map;
  std::unordered_map<std::string, DerivativeCostFunction> cost_derivative_map;
  std::unordered_map<std::string, DerivativeActivationFunction>
      activation_derivative_map;

  void setFunctions(const std::string& cost_func,
                    const std::string& activation_func);

 public:
  FullyConnectedLayer(int num_in_nodes, int num_out_nodes,
                      const std::string& activation_func,
                      const std::string& cost_func);
  // weights and biases
  std::vector<std::vector<double>> weights;
  std::vector<double> biases;

  CostFunction costFunction;
  ActivationFunction activationFunction;
  DerivativeCostFunction derivativeCostFunction;
  DerivativeActivationFunction derivativeActivationFunction;

  // Methods
  std::pair<std::vector<double>, std::vector<double>> calculateOutputs(
      const std::vector<double>& inputs);
};

#endif  // !LAYERS_H
