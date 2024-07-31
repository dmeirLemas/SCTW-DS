# Super Cool And Totally Working Data Science Library (SCTW)

This repository contains a collection of personal machine learning tools and algorithms developed for educational purposes. This is not a professional project and is not intended to be used as such. The code here is intended to provide basic implementations of some common machine learning algorithms and utilities.

## Overview

This project includes the following components:

- **Linear Regression**
- **Multivariate Linear Regression**
- **Polynomial Regression**
- **Train-Test Split**
- **Data Generation Function**
- **Data Cleaner Class**
- **Neural Network and Layer Classes**


## Linear Regression

A simple linear regression implementation that fits a line to the given data points.

### Usage

```python
from SCTW_DS.Regression import LinearRegression

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

model = LinearRegression(x, y)
predictions = model.predict([6, 7, 8])
print(predictions)  # Output: [12.0, 14.0, 16.0]
```
## Multivariate Linear Regression

An implementation of multivariate linear regression that fits a hyperplane to the given data points.

### Usage

```python
from SCTW_DS.Regression import MultivariateLinearRegression

X = [[1, 2], [2, 3], [3, 4], [4, 5]]
Y = [[2], [3], [4], [5]]

model = MultivariateLinearRegression(X, Y)
predictions = model.predict([[5, 6], [6, 7]])
print(predictions)  # Output: [[5.999999999999999], [6.999999999999999]]
```

## Polynomial Regression
An implementation of polynomial regression that fits a polynomial curve to the given data points.

### Usage

```python
from SCTW_DS.Regression import PolynomialRegression

x = [1, 2, 3, 4, 5]
y = [1, 8, 27, 64, 125]

model = PolynomialRegression(x, y, degree=3)
predictions = model.predict([6, 7, 8])
print(predictions)  # Output: [216.0, 343.0, 512.0]
```

## Train-Test Split

A utility function that splits a dataset into training and test sets.

### Usage

```python
import pandas as pd
from SCTW_DS.Utils import trainTestSplit

df = pd.read_csv('your_dataset.csv')
train_data, test_data = trainTestSplit(df, ratio=0.8)
print(len(train_data), len(test_data))  # Output: 80 20 (if there are 100 samples)
```

## Data Generation Function

A function to generate synthetic datasets for testing machine learning algorithms.

### Usage
```python
from SCTW_DS.Utils import generateDataset

def func1(low, high, size):
    return np.random.uniform(low=low, high=high, size=size)

def pois_func(arr):
    return ((arr[1] ** 2 + arr[0] ** 2) > 50**2).astype(float)

generateDataset(
    name="data.csv",
    num_input_variables=2,
    x_funcs=[func1, func1],
    x_func_params=[(0, 100, 1000), (0, 100, 1000)],
    y_func=pois_func,
    balance_dataset=True,
)
```

## dataCleaner

dataCleaner is a Python class designed to help clean and preprocess datasets for better model training and evaluation. It provides functionalities for handling duplicate observations, missing values, and outliers in your training and testing datasets.

### Usage

```python
from SCTW_DS.Utils import dataCleaner  

# Sample data
train_df = pd.DataFrame({
    'A': [1, 2, 2, 4, np.nan],
    'B': [5, np.nan, np.nan, 8, 10],
    'C': [11, 12, 13, 14, 15]
})

test_df = pd.DataFrame({
    'A': [np.nan, 2, 3, 4, 5],
    'B': [5, 6, np.nan, 8, 10],
    'C': [15, 14, 13, 12, 11]
})

# Initialize dataCleaner
cleaner = dataCleaner(train_df, test_df)

# Show duplicate observations
duplicates = cleaner.show_duplicate_observations()
print(duplicates)

# Remove duplicate observations
cleaner.remove_duplicate_observations()

# Show missing values
missing_values = cleaner.show_missing_values()
print(missing_values)

# Fix missing values in column 'A' using the 'Mean' strategy
cleaner.fix_missing_values(feature='A', strategy='Mean')

# Detect outliers in column 'A' using the inter-quartile range strategy
outliers = cleaner.outlier_detection(feature='A')
print(outliers)
```



## NeuralNetwork and Layer

Together form a simple sequential neural network.

### Usage
```python
import numpy as np
import pandas as pd

from SCTW_DS.NeuralNetwork import NeuralNetwork
from SCTW_DS.NeuralNetwork import Layer
from SCTW_DS.Utils import trainTestSplit

# Load the dataset
df = pd.read_csv("data.csv")

# Split the dataset into training and test data points
train, test = trainTestSplit(df, 0.8)

# Define the layers of the neural network
layers = [
    Layer(2, 3, "reLU"),
    Layer(3, 5, "reLU"),
    Layer(5, 3, "reLU"),
    Layer(3, 2, "sigmoid", "mse"),
]

# Initialize the neural network
nn = NeuralNetwork(layers)

# Train the neural network
costs = nn.train(
    iterations=10000,
    data_points=training_data,
    learning_rate=0.01,
    batch_size=50,
    momentum=0.9,
)

model_version = 1
# Save the model
nn.save_model(f"model_{model_version}.pkl")

# Plot the cost over iterations
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost over Iterations")
plt.show()

# Load the model
nn_loaded = NeuralNetwork.load_model(f"model_{model_version}.pkl")

# Verify loaded model
y_pred = nn_loaded.classify(test)

# Calculate accuracy
y_true = [np.argmax(point.expected_outputs) for point in test]
accuracy = sum(y1 == y2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)

print(f"Accuracy: {accuracy * 100:.2f}%")
```

## Note
This project is a personal endeavor and is not intended to be a professional or production-ready implementation. The code is for educational purposes only and may not follow best practices in software development or machine learning.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Installation
You can install the required dependencies using pip:
```sh
pip install SCTW-DS
```























