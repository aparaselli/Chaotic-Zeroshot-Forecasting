# # pipeline_eval.py
# # Step 1: Installation and Imports

# # // pip install chronos-forecasting pandas torch matplotlib

# import pandas as pd
# import numpy as np
# import torch
# from chronos import BaseChronosPipeline
# import matplotlib.pyplot as plt

# # Step 2: Set Device (GPU/CPU)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# # Step 3: Define Evaluation Metrics (WQL and MASE)
# def quantile_loss(y_true, y_pred, alpha):
#     """
#     Computes quantile loss for a given quantile level alpha.
    
#     Parameters:
#     - y_true: True values (numpy array of shape (N,))
#     - y_pred: Predicted quantile values (numpy array of shape (N,))
#     - alpha: Quantile level (scalar)

#     Returns:
#     - Quantile loss (scalar)
#     """
#     diff = y_true - y_pred
#     return np.maximum(alpha * diff, (alpha - 1) * diff).mean()

# def weighted_quantile_loss(y_true, y_pred_quantiles, quantile_levels):
#     """
#     Computes the Weighted Quantile Loss (WQL) with the modified shape of predictions.

#     Parameters:
#     - y_true: True values (numpy array of shape (N,))
#     - y_pred_quantiles: Predicted quantiles (numpy array of shape (K, N), where K is the number of quantiles)
#     - quantile_levels: List or array of quantile levels (K,)

#     Returns:
#     - Weighted Quantile Loss (scalar)
#     """
#     assert y_pred_quantiles.shape[0] == len(quantile_levels)
#     wql_per_quantile = [
#         (2 * quantile_loss(y_true, y_pred_quantiles[j, :], alpha)) / np.abs(y_true).sum()
#         for j, alpha in enumerate(quantile_levels)
#     ]
#     return np.mean(wql_per_quantile)

# def mase(y_true, y_pred, y_past, S):
#     """
#     Computes the Mean Absolute Scaled Error (MASE) based on the provided formula.

#     Parameters:
#     - y_true: Actual values (numpy array of shape (H,))
#     - y_pred: Predicted values (numpy array of shape (H,))
#     - y_past: Historical values (numpy array of shape (C,)) for computing seasonal naive MAE
#     - S: Seasonality parameter (integer)

#     Returns:
#     - MASE score (scalar)
#     """
#     C = len(y_past)
#     H = len(y_true)
#     numerator = np.sum(np.abs(y_pred - y_true)) / H
#     denominator = np.sum(np.abs(y_past[:-S] - y_past[S:])) / (C - S)
#     return numerator / denominator if denominator != 0 else np.inf

# # Step 4: Load Data
# df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
# data_series = torch.tensor(df["#Passengers"].values)

# # Split data for train (context) and test (prediction)
# context, actual = data_series[:-12], data_series[-12:]

# # Step 5: Load Model and Predict
# pipeline = BaseChronosPipeline.from_pretrained(
#     "amazon/chronos-bolt-mini",
#     device_map=device,
#     torch_dtype=torch.bfloat16,
# )

# prediction_length = 12
# forecast = pipeline.predict(
#     context=context, 
#     prediction_length=prediction_length
# )

# # Step 6: Evaluate Predictions
# quantile_levels = np.linspace(0.1, 0.9, 9)
# y_pred_quantiles = forecast.numpy().squeeze()
# y_true = actual.numpy()

# wql_score = weighted_quantile_loss(y_true, y_pred_quantiles, quantile_levels)
# mase_score = mase(y_true, y_pred_quantiles[4, :], context.numpy(), S=12)  # median quantile for MASE

# print(f"Weighted Quantile Loss (WQL): {wql_score:.4f}")
# print(f"Mean Absolute Scaled Error (MASE): {mase_score:.4f}")

# # Step 7: Visualization
# plt.figure(figsize=(12, 6))
# plt.plot(df['Month'], data_series.numpy(), label="Actual")
# plt.plot(df['Month'][-12:], y_pred_quantiles[4, :], label="Predicted Median", linestyle='--')
# plt.fill_between(
#     df['Month'][-12:],
#     y_pred_quantiles[0, :],
#     y_pred_quantiles[-1, :],
#     color='gray', alpha=0.3, label="Prediction Interval"
# )
# plt.xticks(rotation=45)
# plt.xlabel("Month")
# plt.ylabel("Passengers")
# plt.title("Chronos Forecast vs Actual")
# plt.legend()
# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error

class ChronosEvaluator:
    def __init__(self, data_path, test_size=12, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_size = test_size
        self.df = self.load_data(data_path)
        self.context, self.target = self.prepare_data()
        self.results = {}

    def load_data(self, data_path):
        """Load time series data from CSV"""
        df = pd.read_csv(data_path)
        df['Month'] = pd.to_datetime(df['Month'])
        return df.set_index('Month')

    def prepare_data(self):
        """Split data into context and target"""
        series = self.df['#Passengers'].values
        return torch.tensor(series[:-self.test_size]), series[-self.test_size:]

    def quantile_loss(self, y_true, y_pred, alpha):
        """Compute quantile loss for a specific quantile level"""
        diff = y_true - y_pred
        return np.maximum(alpha * diff, (alpha - 1) * diff).mean()

    def weighted_quantile_loss(self, y_true, y_pred_quantiles, quantile_levels):
        """Compute Weighted Quantile Loss (WQL)"""
        assert y_pred_quantiles.shape[0] == len(quantile_levels)
        wql_per_quantile = [
            2 * self.quantile_loss(y_true, y_pred_quantiles[j], alpha) / np.abs(y_true).sum()
            for j, alpha in enumerate(quantile_levels)
        ]
        return np.mean(wql_per_quantile)

    def mase(self, y_true, y_pred, y_past, seasonality=12):
        """Compute Mean Absolute Scaled Error (MASE)"""
        mae = np.mean(np.abs(y_pred - y_true))
        scale = np.mean(np.abs(y_past[seasonality:] - y_past[:-seasonality]))
        return mae / scale if scale != 0 else np.inf

    def evaluate_model(self, model_name):
        """Run inference and compute metrics for a Chronos model"""
        try:
            # Load pretrained model
            pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
            
            # Generate forecasts
            forecast = pipeline.predict(
                context=self.context,
                prediction_length=self.test_size
            )
            
            # Convert to numpy
            quantiles = np.linspace(0.1, 0.9, 9)
            forecast_median = forecast[0][4].numpy()
            forecast_quantiles = forecast[0].numpy()
            
            # Compute metrics
            wql = self.weighted_quantile_loss(self.target, forecast_quantiles, quantiles)
            mase_score = self.mase(self.target, forecast_median, self.context.numpy(), self.test_size)
            
            # Store results
            self.results[model_name] = {
                'forecast': forecast_median,
                'wql': wql,
                'mase': mase_score,
                'quantiles': forecast_quantiles
            }
            
            print(f"Evaluation complete for {model_name}")
            return True
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return False

    def plot_results(self):
        """Plot forecasts and metrics comparison"""
        plt.figure(figsize=(15, 10))
        
        # Plot actual vs forecasts
        plt.subplot(2, 1, 1)
        plt.plot(self.df.index, self.df['#Passengers'], label='Actual')
        for model, result in self.results.items():
            plt.plot(pd.date_range(start=self.df.index[-self.test_size], periods=self.test_size, freq='M'),
                     result['forecast'], '--', label=model)
        plt.title('Actual vs Forecasted Values')
        plt.legend()
        
        # Plot metrics comparison
        plt.subplot(2, 1, 2)
        metrics = ['wql', 'mase']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i+2)
            values = [result[metric] for result in self.results.values()]
            plt.bar(self.results.keys(), values)
            plt.title(f'{metric.upper()} Comparison')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    def run_pipeline(self, model_list):
        """Run full evaluation pipeline"""
        for model in model_list:
            self.evaluate_model(model)
        
        self.plot_results()
        return self.results

if __name__ == "__main__":
    # Example usage
    evaluator = ChronosEvaluator(
        data_path="https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
        test_size=12
    )
    
    # List of Chronos models to evaluate
    model_list = [
        "amazon/chronos-t5-mini",
        "amazon/chronos-t5-large",
        "amazon/chronos-bolt-mini"
    ]
    
    results = evaluator.run_pipeline(model_list)
    
    # Print metric results
    print("\nMetric Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"WQL: {metrics['wql']:.4f}")
        print(f"MASE: {metrics['mase']:.4f}")
