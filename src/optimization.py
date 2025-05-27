import numpy as np  # For numerical operations
from scipy.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize as pymoo_minimize

# --------- Optimization ---------
class Optimization:
    def __init__(self, method="gradient_based", verbose=True):
        self.method = method
        self.verbose = verbose

    def run(self, bounds, func, x0=None):
        if self.method == "gradient_based":
            return self._run_gradient_based(bounds, func, x0)
        elif self.method == "gradient_free":
            return self._run_gradient_free(bounds, func)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _run_gradient_based(self, bounds, func, x0):
        print("\n🔍 Gradient-Based Optimization (SLSQP)...")
        x0 = x0 or np.zeros(len(bounds))
        result = minimize(fun=lambda x: func(np.array([x]))[0, 0], x0=x0, bounds=bounds, method="SLSQP")
        if self.verbose:
            print(f"Min Point: {result.x}, Min Value: {result.fun:.8f}")
        return result

    def _run_gradient_free(self, bounds, func):
        print("\n🔍 Gradient-Free Optimization (NSGA-II)...")

        class CustomProblem(Problem):
            def __init__(self):
                super().__init__(n_var=len(bounds), n_obj=1, xl=np.array([b[0] for b in bounds]),
                                 xu=np.array([b[1] for b in bounds]))

            def _evaluate(self, X, out, **kwargs):
                out["F"] = np.array([func(np.array([x]))[0, 0] for x in X])

        problem = CustomProblem()
        algo = NSGA2(pop_size=50, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                     mutation=PolynomialMutation(eta=20), eliminate_duplicates=True)

        result = pymoo_minimize(problem=problem, algorithm=algo, termination=("n_gen", 50), verbose=self.verbose)
        if self.verbose:
            print(f"Point of Minimum:")
            print(f"{(np.array2string(result.X, formatter={'float_kind': lambda x: '%.8f' % x}))}")

            # Check if result.F is a scalar or a 2D array and handle accordingly
            if np.isscalar(result.F) or len(result.F.shape) == 0:
                print(f"Minimum Value: {result.F:.4e}")
            elif len(result.F.shape) == 1:
                print(f"Minimum Value: {result.F[0]:.4e}")
            else:
                print(f"Minimum Value: {result.F[0][0]:.4e}")
        return result


