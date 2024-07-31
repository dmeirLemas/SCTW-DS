from dataclasses import dataclass, field
from typing import List


@dataclass
class LinearRegression:
    x: List[float]
    y: List[float]
    a: float = field(init=False)
    c: float = field(init=False)

    def __post_init__(self):
        tmp_1 = 0
        n = len(self.x)
        for i in range(n):
            tmp_1 += self.x[i] * self.y[i]

        sum_x = sum(self.x)
        sum_y = sum(self.y)
        tmp_1 -= sum_x * sum_y / n

        tmp_2 = 0
        for i in range(n):
            tmp_2 += self.x[i] * self.x[i]

        tmp_2 -= sum_x * sum_x / n

        self.a = tmp_1 / tmp_2

        self.c = (sum_y - self.a * sum_x) / n

    def _predict(self, x: float) -> float:
        return self.a * x + self.c

    def predict(self, x: List[float]) -> List[float]:
        preds: List[float] = []
        for x_i in x:
            preds.append(self._predict(x_i))

        return preds

    def _se(self, pred: float, y: float) -> float:
        return (pred - y) * (pred - y)

    def mse(self, preds: List[float], y: List[float]) -> float:
        mse = 0
        n = len(preds)
        for i in range(n):
            mse += self._se(preds[i], y[i])
        return mse / n
