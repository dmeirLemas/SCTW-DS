#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../Eigen/Dense"

class FullyConnectedLayer {
 private:
  int num_in_nodes;
  int num_out_nodes;
  std::string activation_func;
  std::string cost_func;

  double mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);
  double cross_entropy(const Eigen::VectorXd& y_true,
                       const Eigen::VectorXd& y_pred);
  Eigen::VectorXd mseDerivative(const Eigen::VectorXd& y_true,
                                const Eigen::VectorXd& y_pred);
  Eigen::VectorXd cross_entropyDerivative(const Eigen::VectorXd& y_true,
                                          const Eigen::VectorXd& y_pred);

  Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
  Eigen::VectorXd relu(const Eigen::VectorXd& x);
  Eigen::VectorXd leaky_relu(const Eigen::VectorXd& x);
  Eigen::VectorXd tanh(const Eigen::VectorXd& x);
  Eigen::VectorXd softmax(const Eigen::VectorXd& x);

  Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& x);
  Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);
  Eigen::VectorXd leaky_relu_derivative(const Eigen::VectorXd& x);
  Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& x);
  Eigen::VectorXd softmax_derivative(const Eigen::VectorXd& x);

  using CostFunction =
      std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using ActivationFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  using DerivativeCostFunction = std::function<Eigen::VectorXd(
      const Eigen::VectorXd&, const Eigen::VectorXd&)>;
  using DerivativeActivationFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

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

  Eigen::MatrixXd weights;
  Eigen::VectorXd biases;

  CostFunction costFunction;
  ActivationFunction activationFunction;
  DerivativeCostFunction derivativeCostFunction;
  DerivativeActivationFunction derivativeActivationFunction;

  std::pair<Eigen::VectorXd, Eigen::VectorXd> calculateOutputs(
      const Eigen::VectorXd& inputs);
};

#endif  // FULLYCONNECTEDLAYER_H
