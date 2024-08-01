# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the path to pybind11
set(CMAKE_PREFIX_PATH "/Users/demirelmas/stuff/random_shit/SCTW-DS_C_REWRITE/env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# Find pybind11 package
find_package(pybind11 REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

include_directories("./src/Eigen") 

# Add the source files
set(SOURCES
    src/NeuralNetwork/cpp/FullyConnectedLayer.cpp
    src/NeuralNetwork/cpp/Network.cpp
)

# Create the module
pybind11_add_module(NeuralNetwork ${SOURCES} neural_network.cpp)

# Include directories
target_include_directories(NeuralNetwork PRIVATE src/NeuralNetwork/header)

# Link OpenMP libraries
target_link_libraries(NeuralNetwork PRIVATE OpenMP::OpenMP_CXX)
