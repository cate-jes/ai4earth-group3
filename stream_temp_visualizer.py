import numpy as np
import matplotlib.pyplot as plt

class StreamTempVisualizer:
    def __init__(self, denorm_mean=None, denorm_std=None):
        # Initialize lists to store predictions, ground truths, and persistence baseline
        self.predictions = []
        self.ground_truths = []
        self.persistence_baseline = []
        # Store mean and std for denormalization if provided
        self.denorm_mean = denorm_mean
        self.denorm_std = denorm_std

    def add_batch(self, y_pred, y_true, last_input_temp):
        """
        Add a batch of predictions and ground truths.
        y_pred, y_true: shape (7,)
        last_input_temp: scalar (used for persistence)
        """
        # Denormalize if mean and std are provided
        if self.denorm_mean is not None and self.denorm_std is not None:
            y_pred = y_pred * self.denorm_std + self.denorm_mean
            y_true = y_true * self.denorm_std + self.denorm_mean
            last_input_temp = last_input_temp * self.denorm_std + self.denorm_mean

        # Create persistence baseline by repeating last input temperature
        persistence = np.repeat(last_input_temp, len(y_true))

        # Store the batch results
        self.predictions.append(y_pred)
        self.ground_truths.append(y_true)
        self.persistence_baseline.append(persistence)

    def compute_metrics(self):
        # Concatenate all batches for evaluation
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.ground_truths)
        y_persist = np.concatenate(self.persistence_baseline)

        # Compute and return RMSE and MAE for model and persistence baseline
        return {
            "Model RMSE": np.sqrt(self.mean_squared_error(y_true, y_pred)),
            "Persistence RMSE": np.sqrt(self.mean_squared_error(y_true, y_persist)),
            "Model MAE": self.mean_absolute_error(y_true, y_pred),
            "Persistence MAE": self.mean_absolute_error(y_true, y_persist),
        }

    def plot_example(self, index=0):
        """
        Plot one sequence: model vs truth vs persistence
        """
        # Select the sequence to plot
        y_pred = self.predictions[index]
        y_true = self.ground_truths[index]
        y_persist = self.persistence_baseline[index]

        # Plot ground truth, model prediction, and persistence baseline
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, 8), y_true, label='Ground Truth', marker='o')
        plt.plot(range(1, 8), y_pred, label='Model Prediction', marker='x')
        plt.plot(range(1, 8), y_persist, label='Persistence', linestyle='--')
        plt.xlabel('Forecast Day')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Prediction vs Truth (Sequence {index})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        # Compute mean squared error between true and predicted values
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        # Compute mean absolute error between true and predicted values
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))
    
def test_model(model, dataloader, visualizer, autoreg_idx):
    """
    Evaluate model on given dataloader using StreamTempVisualizer.

    Args:
        model: Trained PyTorch model (LSTM)
        dataloader: torch.utils.data.DataLoader returning (X, y)
        visualizer: Instance of StreamTempVisualizer
        autoreg_idx: Index of the stream temp in the input feature vector
    """
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch.float32)  # Shape: (batch, 30, num_features)
            y = y.to(torch.float32)  # Shape: (batch, 7)

            preds = model(X)  # Output shape: (batch, 7)

            for i in range(X.shape[0]):
                last_input_temp = X[i, -1, autoreg_idx].item()
                y_pred_np = preds[i].cpu().numpy()
                y_true_np = y[i].cpu().numpy()
                visualizer.add_batch(y_pred_np, y_true_np, last_input_temp)

