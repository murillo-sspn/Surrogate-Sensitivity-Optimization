import numpy as np  # For numerical operations
from scipy.stats.qmc import LatinHypercube, Sobol  # For Latin Hypercube Sampling and Sobol Sampling

# --------- SamplingHandler ---------
class SamplingHandler:
    def __init__(self, sampler_method, bounds, n_samples, random_seed=None):
        self.sampler_method = sampler_method
        self.bounds = np.array(bounds)
        self.n_design_variables = np.shape(self.bounds)[0]
        self.n_samples = int(np.ceil(n_samples))
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        self.set_sample()
        print("\n📈 Sampling Metrics:")
        print(f"n_samples: {self.n_samples:.0f}, n_design_variables: {self.n_design_variables:.0f}")
        print(f"bounds:")
        print(np.array2string(self.bounds, formatter={'float_kind': lambda x: "%.8f" % x}))

    def set_sample(self):

        if self.sampler_method == 'uniform':
            self.sample = self.uniform()
        elif self.sampler_method == 'lhs':
            self.sample = self.lhs()
        elif self.sampler_method == 'sobol':
            self.sample = self.sobol()
        elif self.sampler_method == 'gaussian':
            self.sample = self.gaussian()
        else:
            raise ValueError(f"Unknown sampler method: {self.sampler_method}")

    def uniform(self):
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_samples, self.bounds.shape[0]))

    def lhs(self):
        lhs_sampler = LatinHypercube(d=self.bounds.shape[0], seed=self.random_seed)
        samples = lhs_sampler.random(n=self.n_samples)
        return self._scale_to_bounds(samples)

    def sobol(self):
        m = int(np.log2(self.n_samples))
        if 2 ** m != self.n_samples:
            raise ValueError("Sobol requires n_samples to be a power of 2.")
        sobol_sampler = Sobol(d=self.bounds.shape[0], seed=self.random_seed)
        samples = sobol_sampler.random_base2(m=m)
        return self._scale_to_bounds(samples)

    def gaussian(self):
        means = self.bounds.mean(axis=1)
        std_dev = (self.bounds[:, 1] - self.bounds[:, 0]) / 6
        samples = np.random.normal(loc=means, scale=std_dev, size=(self.n_samples, self.bounds.shape[0]))
        return np.clip(samples, self.bounds[:, 0], self.bounds[:, 1])

    def _scale_to_bounds(self, samples):
        return samples * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

