import numpy as np  # For numerical operations like sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # For evaluation metrics
from smt.surrogate_models import KRG, RBF  # For surrogate models
from src.plot_exporter import PlotExporter  # For exporting plots

# --------- Surrogate Model Wrapper ---------
class SurrogateModel:
    def __init__(self, model_type='kriging', output_dir='outputs', verbose=True):
        self.model_type = model_type.lower()
        self.verbose = verbose
        self.plot_exporter = PlotExporter(output_dir=output_dir, verbose=verbose)
        self.model = self._create_model()

    def _create_model(self):
        if self.model_type == 'kriging':
            return KRG(print_global=False)
        elif self.model_type == 'rbf':
            return RBF(print_global=False)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X, y):
        self.model.set_training_values(X, y)
        self.model.train()

    def predict(self, X):
        return self.model.predict_values(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X).ravel()
        y_true = y_true.ravel()
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        print("\n📈 Evaluation Metrics:")
        print(f"R²: {r2:.16f}, RMSE: {rmse:.16f}, MAE: {mae:.16f}")
        self.plot_exporter.predicted_vs_actual(y_true, y_pred)
        self.plot_exporter.plot_residuals(y_true, y_pred)

