import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting results of Sobol sensitivity

from SALib.sample import saltelli, morris as morris_sample  # For sampling points for Sobol and Morris analysis
from SALib.analyze import sobol, morris as morris_analyze  # For sensitivity analysis (Sobol & Morris)

from src.plot_exporter import PlotExporter  # For exporting plots



# --------- Sensitivity Analysis ---------
class SensitivityAnalysis:
    def __init__(self, func, var_ranges, output_dir='outputs', N=1000, method='morris', verbose=True):
        """
        Unified Sensitivity Analysis class.

        Parameters:
        - func (callable): Function for which sensitivity needs to be evaluated.
        - var_ranges (list): List of bounds for each variable as [[lower, upper], ...].
        - output_dir (str): Directory where plots should be saved.
        - N (int): Number of samples to use for sensitivity analysis.
        - method (str): Sensitivity analysis method ('morris', 'sobol', 'ssanova').
        - verbose (bool): Whether to print detailed logs and show plots.
        """
        self.func = func
        self.var_ranges = var_ranges
        self.problem = {
            'num_vars': len(var_ranges),
            'names': [f'x{i + 1}' for i in range(len(var_ranges))],
            'bounds': var_ranges
        }
        self.plot_exporter = PlotExporter(output_dir=output_dir, verbose=verbose)
        self.N = N
        self.method = method.lower()
        self.verbose = verbose

    def analyze(self):
        """
        Perform sensitivity analysis using the selected method.
        """
        if self.method == 'morris':
            self._analyze_morris()
        elif self.method == 'sobol':
            self._analyze_sobol()
        elif self.method == 'ssanova':
            self._analyze_ssanova()
        else:
            raise ValueError(f"Unknown sensitivity method: {self.method}")

    def _analyze_morris(self):
        """Perform Morris' Method sensitivity analysis."""
        print("\n🔍 Morris Sensitivity Analysis...")
        param_values = morris_sample.sample(self.problem, N=self.N, num_levels=4, grid_jump=2)
        Y = self.func(param_values).ravel()
        results = morris_analyze.analyze(self.problem, param_values, Y, print_to_console=False)
        for name, mu_star, sigma in zip(self.problem['names'], results['mu_star'], results['sigma']):
            print(f"{name}: μ* = {mu_star:.8f}, σ = {sigma:.8f}")
        self.plot_exporter.morris_sensitivity(results['mu_star'], results['sigma'], self.problem['names'])

    def _analyze_sobol(self):
        """Perform Sobol sensitivity analysis."""
        print("\n🔍 Sobol Sensitivity Analysis...")
        param_values = saltelli.sample(self.problem, N=self.N, calc_second_order=True)
        Y = self.func(param_values).ravel()
        results = sobol.analyze(self.problem, Y, print_to_console=False)
        print("\n-- Sobol Sensitivity Results --")
        for name, s1, st in zip(self.problem['names'], results['S1'], results['ST']):
            print(f"{name}: S1 = {s1:.8f}, ST = {st:.8f}")
        self._sobol_plot(results['S1'], results['ST'], self.problem['names'])

    def _analyze_ssanova(self):
        """Perform SS-ANOVA sensitivity analysis."""
        print("\n🔍 SS-ANOVA Sensitivity Analysis...")
        # A placeholder for future implementation of SS-ANOVA. Depending on the library,
        # you can implement this here and output visualizations.
        print("SS-ANOVA is currently not implemented.")
        raise NotImplementedError("SS-ANOVA sensitivity analysis is not yet implemented!")

    def _sobol_plot(self, S1, ST, var_names):
        """Plot Sobol sensitivity results."""
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(var_names))
        ax.bar(x - 0.2, S1, 0.4, label="First-order (S1)", color='dodgerblue')
        ax.bar(x + 0.2, ST, 0.4, label="Total-order (ST)", color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(var_names, rotation=45)
        ax.set_ylabel("Sensitivity Index")
        ax.set_title("Sobol Sensitivity Analysis")
        ax.legend()
        ax.grid(True)
        self.plot_exporter.save_plot(fig, "sobol_sensitivity")

