import configparser  # For parsing the .cfg configuration file
import numpy as np  # For numerical operations
from src.sampling import SamplingHandler  # Custom handler for sampling methods
from src.test_functions import TestFunctions  # Custom test functions (e.g., Rosenbrock)
from src.surrogate_model import SurrogateModel  # Custom class for surrogate model fitting and prediction
from src.sensitivity_analysis import SensitivityAnalysis  # Sensitivity analysis methods (e.g., Morris)
from src.optimization import Optimization  # Optimization algorithms (e.g., gradient-based, gradient-free)
from src.plot_exporter import PlotExporter  # For exporting plots

# --------- Benchmark Experiment ---------
class Experiment:
    def __init__(self, config_file):

        # Parse the .cfg file
        config = configparser.ConfigParser()
        config.read(config_file)

        # Experiment settings
        model_type = config.get('Experiment', 'model_type', fallback='kriging')
        test_function = config.get('Experiment', 'test_function', fallback='rosenbrock')
        sensitivity_method = config.get('Experiment', 'sensitivity_method', fallback='morris')
        optimizer = config.get('Experiment', 'optimizer', fallback='gradient_based')
        output_dir = config.get('Experiment', 'output_dir', fallback='outputs')

        # Sampling settings
        bounds = eval(config.get('Sampling', 'bounds', fallback='[[-2, 2], [-1, 3]]'))
        n_samples = config.getint('Sampling', 'n_samples', fallback=100)
        sampler_train = config.get('Sampling', 'sampler_train', fallback='lhs')
        sampler_test = config.get('Sampling', 'sampler_test', fallback='lhs')
        test_to_train_ratio = config.getfloat('Sampling', 'test_to_train_ratio', fallback=0.10)

        # Initialize
        self.model = SurrogateModel(model_type=model_type, output_dir=output_dir)
        self.bounds = bounds
        self.n_samples = n_samples
        self.output_dir = output_dir
        self.sampler_method_train = sampler_train
        self.sampler_method_test = sampler_test
        self.sampler_train = SamplingHandler(self.sampler_method_train,
                                             bounds=self.bounds,
                                             n_samples=self.n_samples)
        self.sampler_test = SamplingHandler(self.sampler_method_test,
                                            bounds=self.bounds,
                                            n_samples=test_to_train_ratio*self.n_samples)
        self.test_function = TestFunctions(test_function_name=test_function)
        self.sensitivity_method = sensitivity_method
        self.optimizer_method = optimizer
        self.optimizer = Optimization(method=self.optimizer_method)
        self.plot_exporter = PlotExporter(output_dir=output_dir)

    def run(self):

        # Train model
        # Generate the samples
        X_train = self.sampler_train.sample
        # Plot the samples
        self.plot_exporter.plot_samples(X_train=X_train, sampler_method_train=self.sampler_method_train)
        # Get training data results
        Y_train =  self.test_function.test_function(X_train)
        # Fit model
        self.model.fit(X_train, Y_train)

        # Test model
        # Generate the samples
        X_test = self.sampler_test.sample
        # Plot the samples
        self.plot_exporter.plot_samples(X_train=X_train, X_test=X_test, sampler_method_train=self.sampler_method_train, sampler_method_test=self.sampler_method_test)
        # Get testing data results
        Y_test = self.test_function.test_function(X_test)
        # Evaluate model at points
        self.model.evaluate(X_test, Y_test)

        # Sensitivity Analysis
        sensitivity = SensitivityAnalysis(func=self.model.predict, var_ranges=self.bounds,
                                          output_dir=self.output_dir, method=self.sensitivity_method, N=1000)
        sensitivity.analyze()

        # Optimization
        self.optimizer.run(self.bounds, func=self.model.predict)

