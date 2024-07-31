#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string>
#include <vector>

#include "Layers.h"

class NeuralNetwork {
 public:
  NeuralNetwork(const std::vector<FullyConnectedLayer>& layers);

  std::vector<double> calculateOutputs(const std::vector<double>& inputs);
  int classify(const std::vector<double>& inputs);
  double cost(
      const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
          data_points);
  void learn(
      const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
          training_data,
      double learning_rate, int batch_size, double momentum);
  std::vector<double> train(
      int iterations,
      const std::vector<std::pair<std::vector<double>, std::vector<double>>>&
          data_points,
      double learning_rate, int batch_size, double momentum);
  void saveModel(const std::string& file_path);
  static NeuralNetwork loadModel(const std::string& file_path);

 private:
  std::vector<FullyConnectedLayer> layers;
  std::vector<std::vector<std::vector<double>>> velocity_w;
  std::vector<std::vector<double>> velocity_b;
};

#endif  // NEURALNETWORK_H
