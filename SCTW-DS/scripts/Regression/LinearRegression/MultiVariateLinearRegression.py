from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MultivariateLinearRegression:
    X: List[List[float]]  # Each inner list is a feature vector
    Y: List[List[float]]  # Each inner list is a target vector

    def __post_init__(self):
        self.x = np.array(self.X)
        self.y = np.array(self.Y)

        # Add a column of ones to X for the intercept term
        X_with_intercept = np.hstack((np.ones((self.x.shape[0], 1)), self.x))

        # Compute the coefficients using the Normal Equation: (X^T X)^-1 X^T Y
        X_transpose = X_with_intercept.T
        self.coefficients = (
            np.linalg.pinv(X_transpose @ X_with_intercept) @ X_transpose @ self.y
        )

    def _predict(self, X: List[float]) -> List[float]:
        X_with_intercept = np.hstack(([1], X))
        return np.dot(X_with_intercept, self.coefficients).tolist()

    def predict(self, X: List[List[float]]) -> List[List[float]]:
        return [self._predict(x_i) for x_i in X]

    def _se(self, pred: float, y: float) -> float:
        return (pred - y) * (pred - y)

    def mse(self, preds: List[float], y: List[float]) -> float:
        mse = 0
        n = len(preds)
        for i in range(n):
            mse += self._se(preds[i], y[i])
        return mse / n
