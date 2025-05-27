import os  # For creating directories and managing file paths
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting and visualizations

# --------- Plot Exporter ---------
class PlotExporter:
    def __init__(self, output_dir='outputs', verbose=True):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        file_path = os.path.join(self.output_dir, f"{filename}.pdf")
        fig.savefig(file_path, bbox_inches='tight')
        if self.verbose:
            plt.show()
        plt.close(fig)

    def plot_samples(self, X_train=None, X_test=None, sampler_method_train=None, sampler_method_test=None):
        """
        Plot the sampled points in the design space.

        Parameters:
            X_train (np.ndarray): The sampled data points (n_samples x n_dimensions).
        """

        if X_train.shape[1] == 2:  # Only plot if there are exactly 2 dimensions
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.7, label="Training Data")
            if X_test is not None:
                ax.scatter(X_test[:, 0], X_test[:, 1], c='red', alpha=0.7, label="Testing Data")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            if sampler_method_test is None:
                title = f"Sampled Points - {sampler_method_train.title()} Sampling"
                filename = f"sampled_points_{sampler_method_train}"
            else:
                title = f"Sampled Points - {sampler_method_train.title()}/{sampler_method_test.title()} Samplings"
                filename = f"sampled_points_{sampler_method_train}_{sampler_method_test}"
            ax.legend()
            ax.set_title(title)
            ax.grid(True)
            fig.tight_layout()
            self.save_plot(fig, filename)


    def predicted_vs_actual(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, y_pred, c='blue', alpha=0.7)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predicted vs True")
        ax.grid(True)
        fig.tight_layout()
        self.save_plot(fig, "predicted_vs_true")

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true, residuals, c='red', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        ax.grid(True)
        fig.tight_layout()
        self.save_plot(fig, "residuals_plot")

    def morris_sensitivity(self, mu_star, sigma, var_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(mu_star, sigma, c='dodgerblue', s=100, alpha=0.7)
        for i, txt in enumerate(var_names):
            ax.text(mu_star[i] + 0.01, sigma[i] + 0.01, txt, fontsize=9)
        ax.set_xlabel(r"$\mu^*$ (Mean of absolute EE)")
        ax.set_ylabel(r"$\sigma$ (Standard deviation of EE)")
        ax.set_title("Morris Sensitivity Analysis")
        ax.grid(True)
        self.save_plot(fig, "morris_sensitivity")

