import numpy as np

# --------- Test Functions ---------
class TestFunctions:

    def __init__(self, test_function_name=None):
        self.test_function_name = test_function_name
        self.test_function = self.get_test_function()

    def get_test_function(self):
        if self.test_function_name == 'rosenbrock':
            return TestFunctions.rosenbrock
        elif self.test_function_name == 'branin':
            return TestFunctions.branin
        elif self.test_function_name == 'himmelblau':
            return TestFunctions.himmelblau
        elif self.test_function_name == 'ackley':
            return TestFunctions.ackley
        elif self.test_function_name == 'rastrigin':
            return TestFunctions.rastrigin
        else:
            raise ValueError(f"Unknown test function: {self.test_function_name}")


    @staticmethod
    def rosenbrock(X):
        """
        Rosenbrock function.
        Formula: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
        """
        x = X[:, 0]
        y = X[:, 1]
        return ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2).reshape(-1, 1)

    @staticmethod
    def branin(X):
        """
        Branin function.
        Formula: f(x, y) = (y - (5.1/(4*pi^2)) * x^2 + (5/pi)*x - 6)^2 + 10*(1 - 1/(8*pi))*cos(x) + 10
        """
        x = X[:, 0]
        y = X[:, 1]
        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        d = 6
        e = 10
        f = 1 / (8 * np.pi)
        return (a * (y - b * x ** 2 + c * x - d) ** 2 + e * (1 - f) * np.cos(x) + e).reshape(-1, 1)

    @staticmethod
    def himmelblau(X):
        """
        Himmelblau function.
        Formula: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        """
        x = X[:, 0]
        y = X[:, 1]
        return ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2).reshape(-1, 1)

    @staticmethod
    def ackley(X):
        """
        Ackley function.
        Formula:
        f(x, y) = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2))) - exp(0.5 * (cos(2*pi*x) + cos(2*pi*y))) + e + 20
        """
        x = X[:, 0]
        y = X[:, 1]
        part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
        part2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        return (part1 + part2 + np.e + 20).reshape(-1, 1)

    @staticmethod
    def rastrigin(X):
        """
        Rastrigin function.
        Formula: f(x, y) = 10 * n + sum(x_i^2 - 10 * cos(2 * pi * x_i)) for i = 1...n
        """
        n = X.shape[1]
        A = 10
        return (A * n + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)).reshape(-1, 1)

