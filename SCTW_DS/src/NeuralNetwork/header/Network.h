#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

#include "../../Eigen/Dense"
#include "FullyConnectedLayer.h"

class NeuralNetwork {
 public:
  NeuralNetwork(const std::vector<FullyConnectedLayer>& layers);

  std::vector<double> classifyAll(
      const std::vector<std::vector<double>>& inputs);
  Eigen::VectorXd calculateOutputs(const Eigen::VectorXd& input);
  double cost(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
                  data_points);
  void learn(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
                 training_data,
             double learning_rate, int batch_size);
  std::vector<double> train(
      int iterations,
      const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
          data_points,
      double learning_rate, int batch_size);
  void saveModel(const std::string& file_path);
  static NeuralNetwork loadModel(const std::string& file_path);

 private:
  double classify(const Eigen::VectorXd& input);
  std::vector<FullyConnectedLayer> layers;
  std::vector<Eigen::MatrixXd> velocity_w;
  std::vector<Eigen::VectorXd> velocity_b;
  std::vector<Eigen::MatrixXd> m_w;
  std::vector<Eigen::MatrixXd> v_w;
  std::vector<Eigen::VectorXd> m_b;
  std::vector<Eigen::VectorXd> v_b;
  double beta1;
  double beta2;
  double epsilon;
  int iterations;
};

#endif  // NEURALNETWORK_H
