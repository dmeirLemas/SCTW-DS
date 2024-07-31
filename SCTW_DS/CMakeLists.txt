cmake_minimum_required(VERSION 3.12)
project(NeuralNetwork)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Specify the path to pybind11
set(CMAKE_PREFIX_PATH "/Users/demirelmas/stuff/random_shit/SCTW-DS_C_REWRITE/env/lib/python3.12/site-packages")

find_package(pybind11 REQUIRED)

# Add the source files
set(SOURCES
    src/NeuralNetwork/cpp/Layers.cpp
    src/NeuralNetwork/cpp/Network.cpp
)

# Create the module
pybind11_add_module(NeuralNetwork ${SOURCES})

# Include directories
target_include_directories(NeuralNetwork PRIVATE header)
