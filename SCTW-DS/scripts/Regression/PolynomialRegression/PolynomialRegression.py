from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class PolynomialRegression:
    X: List[float]
    Y: List[float]

    degree: int

    def __post_init__(self):
        x = np.array(self.X)
        y = np.array(self.Y)

        # Add a column of ones to X for the intercept term
        X = np.vander(x, self.degree + 1)
        X_Transpose = X.T
        self.coefficients = np.linalg.pinv(X_Transpose @ X) @ X_Transpose @ y

    def _predict(self, X: float) -> float:
        X_design = np.vander([X], self.degree + 1)
        return np.dot(X_design, self.coefficients).tolist()[0]

    def predict(self, X: List[float]) -> List[float]:
        return [self._predict(x_i) for x_i in X]

    def _se(self, pred: float, y: float) -> float:
        return (pred - y) * (pred - y)

    def mse(self, preds: List[float], y: List[float]) -> float:
        mse = 0
        n = len(preds)
        for i in range(n):
            mse += self._se(preds[i], y[i])
        return mse / n
