"""
Linear Regression
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.b0 = None
        self.b1 = None
        self.X = None
        self.Y = None
        self.is_predictable = False
        self.mean_x = None
        self.mean_y = None

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.mean_x = np.mean(self.X)
        self.mean_y = np.mean(self.Y)
        self.b1 = sum(np.multiply(self.X - self.mean_x, self.Y - self.mean_y)) / sum((self.X - self.mean_x) ** 2)
        self.b0 = self.mean_y - self.b1 * self.mean_x
        if 0 <= self.error() <= 1:
            self.is_predictable = True

    def error(self):
        y_bar = self.X * self.b1 + self.b0
        print(y_bar)
        error = sum((y_bar - self.mean_y) ** 2) / sum((self.Y - self.mean_y) ** 2)
        # print(error)
        return error

    def predict(self, x):
        if self.is_predictable:
            print(f"y={self.b0}+{self.b1}*X")
            return (self.b0 + self.b1 * np.array(x))
        else:
            return -1


if __name__ == '__main__':
    ob = LinearRegression()
    x = np.linspace(0, 50)
    y = np.linspace(1, 25)
    ob.fit(x, y)
    predicted_y = ob.predict(x)
    print(predicted_y)
    plt.style.use('seaborn')
    plt.scatter(x, y)
    plt.plot(x, predicted_y)
    plt.tight_layout()
    plt.show()
