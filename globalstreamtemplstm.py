import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import tqdm
from typing import Tuple

import global_data_processing

SITE_IDS = [1450, 1565, 1571, 1573, 1641]
RESULTS_DIR = 'results/global'

seed = 42
torch.manual_seed(seed)
    # If using CUDA, also set the CUDA seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) # for a single GPU
    # or torch.cuda.manual_seed_all(seed) # for all GPUs

#todo: have reshape either take in 30 days for 1 day of output and then use that output to calculate 6 more days for forecast. have it mask out the next 7 days in training so the model doesnt use it to ensure no data leakage. in forecast have a reshape to output 8 values so take segement 30 and output forecast 31-37, y = forecast_length +1, x = seq_length + target_len+ forecast_length, _), like x, y = new_reshape for i in forecast_len+1, predict 0^k forecast window_x=x[i:i+seq_len] window_y = y[i] out = model.forward(window_x), pick masked values and replace with model rollout prediction have still normalized and stored, calc rmse over timestep 

# Utility functions
def calc_rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    mask = (obs >= 0) & ~np.isnan(obs)
    o, s = obs[mask], sim[mask]
    return float(np.sqrt(np.mean((s - o) ** 2)))

def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    loss_func,
    epoch: int,
    device: torch.device
):
    """Train for one epoch with progress bar"""
    model.train()
    pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    total_loss = 0.0

    for xs, ys in pbar:
        xs, ys = xs.to(device), ys.to(device)
        optimizer.zero_grad()
        y_hat = model(xs)
        loss = torch.sqrt(loss_func(y_hat, ys))  # RMSE loss
        total_loss += loss.item()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model and return predictions and observations"""
    model.eval()
    obs_list, pred_list = [], []
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(device)
            y_hat = model(xs)
            obs_list.append(ys)
            pred_list.append(y_hat.cpu())
    obs = torch.cat(obs_list).cpu()
    preds = torch.cat(pred_list).cpu()
    return obs, preds

# Utility: reshape data for LSTM (original 2-value return for compatibility)
def reshape_data(x, y, seq_length):
    """Reshape data for LSTM input - improved version from streamtemplstm"""
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, n_features = x.shape
    n_sequences = n_samples - seq_length - 1

    x_seq = np.zeros((n_sequences, seq_length, n_features), dtype=x.dtype)
    y_seq = np.zeros((n_sequences, 1), dtype=y.dtype)

    for i in range(n_sequences):
        x_seq[i] = x[i : i + seq_length]
        y_seq[i] = y[i + seq_length]  # Use next timestep, not current

    return x_seq, y_seq

# Plotting function
def plot_results(date_range, observations, predictions, title, rmse, site, persistence_predictions=None, persistence_rmse=None):  
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(date_range, observations, label="Observation", linewidth=2, linestyle='-', marker='o', markersize=3, color='black')
    ax.plot(date_range, predictions, label=f"LSTM Prediction (RMSE: {rmse:.3f})", linewidth=2, linestyle='--', marker='x', markersize=3, color='blue')
    
    # Add persistence baseline if provided
    if persistence_predictions is not None:
        ax.plot(date_range, persistence_predictions, label=f"Persistence Baseline (RMSE: {persistence_rmse:.3f})", 
                linewidth=2, linestyle=':', marker='s', markersize=2, color='red', alpha=0.7)
    
    ax.legend(fontsize='large')
    ax.set_title(f"{title} - LSTM vs Persistence Comparison", fontsize=16)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Temperature (°C)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Add improvement text if persistence is available
    if persistence_rmse is not None:
        improvement = ((persistence_rmse - rmse) / persistence_rmse) * 100
        ax.text(0.02, 0.98, f'LSTM Improvement: {improvement:.1f}%', 
                transform=ax.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"site_{site}_forecast.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filepath}")
    
    # plt.show()

class GlobalStreamTempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, y_scaler=None, static_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.static_size = static_size
        
        lstm_input_size = input_size + static_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # Store the y_scaler for denormalization of predictions
        self.y_scaler = y_scaler
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier normal initialization"""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
    def denormalize_predictions(self, normalized_data):
        """Convert normalized predictions/targets back to original temperature scale"""
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(normalized_data.reshape(-1, 1)).squeeze()
        return normalized_data
        
    def forward(self, x, static=None):
        # Handle static features if provided
        if self.static_size > 0 and static is not None:
            seq_len = x.size(1)
            static_rep = static.unsqueeze(1).repeat(1, seq_len, 1)
            x = torch.cat([x, static_rep], dim=-1)
            
        out, (h_n, _) = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out)
        # Take the last timestep output
        out = out[:, -1]
        return out

    def fit(self, train_loader, optimizer, loss_func, device, n_epochs=30, test_loader=None):
        """Training method with RMSE loss and progress bars"""
        self.train()
        
        # Store RMSE values for plotting
        train_rmse_history = []
        test_rmse_history = []
        
        # Debug: Check training data for NaNs/Infs
        print("Checking training data for NaNs/Infs...")
        for i, (Xb, Yb) in enumerate(train_loader):
            if torch.isnan(Xb).any() or torch.isinf(Xb).any():
                print(f"NaN/Inf found in training X at batch {i}")
            if torch.isnan(Yb).any() or torch.isinf(Yb).any():
                print(f"NaN/Inf found in training Y at batch {i}")
            if i >= 5:  # Only check first few batches
                break
        print("Training data check complete.")
        
        for epoch in range(1, n_epochs+1):
            # Clear GPU cache before each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            # Training phase
            train_rmse = train_epoch(self, optimizer, train_loader, loss_func, epoch, device)
            train_rmse_history.append(train_rmse)
            
            # Evaluation phase
            if test_loader is not None:
                obs, preds = eval_model(self, test_loader, device)
                test_rmse = torch.sqrt(loss_func(preds, obs)).item()
                test_rmse_history.append(test_rmse)
                
                tqdm.tqdm.write(f"Epoch {epoch:02d} — Train RMSE: {train_rmse:.3f} - Test RMSE: {test_rmse:.3f}")
            else:
                tqdm.tqdm.write(f"Epoch {epoch:02d} — Train RMSE: {train_rmse:.3f}")
        
        # Plot RMSE comparison if we have both train and test data
        if test_loader is not None and train_rmse_history:
            self._plot_training_progress(train_rmse_history, test_rmse_history)
        
        return train_rmse_history
    
    def _calculate_persistence_rmse(self, test_loader, device):
        """Calculate RMSE for persistence baseline using sequence-level approach"""
        self.eval()
        persistence_errors = []
        
        with torch.no_grad():
            for Xb, Yb in test_loader:
                # For each sequence, use the last temperature value as persistence prediction
                persistence_pred = Xb[:, -1, -1]  # Last timestep, last feature (temperature)
                actual = Yb.squeeze()
                
                # Calculate squared errors
                errors = (persistence_pred - actual) ** 2
                persistence_errors.extend(errors.cpu().numpy())
        
        self.train()  # Return to training mode
        return np.sqrt(np.mean(persistence_errors))
    
    def _plot_rmse_comparison(self, model_rmse, persistence_rmse):
        """Plot model RMSE vs persistence RMSE over epochs"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(model_rmse) + 1)
        
        plt.plot(epochs, model_rmse, 'b-', label='Model RMSE', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, persistence_rmse, 'r--', label='Persistence Baseline RMSE', linewidth=2, marker='s', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Model Performance vs Persistence Baseline', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add improvement percentage text
        final_improvement = ((persistence_rmse[-1] - model_rmse[-1]) / persistence_rmse[-1]) * 100
        plt.text(0.7, 0.95, f'Final Improvement: {final_improvement:.1f}%', 
                transform=plt.gca().transAxes, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"rmse_comparison.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"RMSE comparison plot saved: {filepath}")
        
        # plt.show()
    
    def _plot_training_progress(self, train_rmse, test_rmse):
        """Plot training progress"""
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(train_rmse) + 1)
        
        plt.plot(epochs, train_rmse, 'b-', label='Train RMSE', linewidth=2, marker='o', markersize=4)
        plt.plot(epochs, test_rmse, 'r--', label='Test RMSE', linewidth=2, marker='s', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Training Progress: Train vs Test RMSE', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add final performance text
        final_train = train_rmse[-1]
        final_test = test_rmse[-1]
        plt.text(0.7, 0.95, f'Final Train RMSE: {final_train:.3f}\nFinal Test RMSE: {final_test:.3f}', 
                transform=plt.gca().transAxes, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"training_progress.png"
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved: {filepath}")
        
        # plt.show()
           
    def _calculate_persistence_forecast(self, observations):
        """Calculate persistence baseline predictions (using previous day's temperature)"""
        # For persistence, we predict each day's temperature as the previous day's temperature
        # For the first day, we use the first observation as the prediction
        persistence_preds = np.zeros_like(observations)
        persistence_preds[0] = observations[0]  # First prediction is the first observation
        persistence_preds[1:] = observations[:-1]  # Shift observations by one day
        return persistence_preds
    
    def evaluate(self, test_loader, device, date_range=None, title="Global Model", site="global"):
        """Evaluate model and return denormalized predictions and observations"""
        self.eval()
        obs_list, pred_list = [], []
        with torch.no_grad():
            for xs, ys in test_loader:
                xs = xs.to(device)
                y_hat = self(xs)
                obs_list.append(ys)
                pred_list.append(y_hat.cpu())
        
        obs = torch.cat(obs_list).cpu()
        preds = torch.cat(pred_list).cpu()
        
        # Calculate RMSE
        obs_np = obs.numpy()
        preds_np = preds.numpy()
        rmse = np.sqrt(np.mean((preds_np - obs_np) ** 2))
        print(f"Test RMSE: {rmse:.3f}")
        
        preds_denorm = self.denormalize_predictions(preds_np)
        obs_denorm = self.denormalize_predictions(obs_np)
        
        if date_range is not None:
            # Calculate persistence baseline for comparison
            persistence_preds_denorm = self._calculate_persistence_forecast(obs_denorm)
            persistence_rmse = np.sqrt(np.mean((persistence_preds_denorm - obs_denorm) ** 2))
            
            # Plot denormalized values with persistence comparison
            plot_results(date_range, obs_denorm, preds_denorm, title=title, rmse=rmse, site=site,
                        persistence_predictions=persistence_preds_denorm, persistence_rmse=persistence_rmse)
        return preds_denorm, obs_denorm, rmse

    def forecast(self, fc_X, fc_Y, seq_len, batch_size, device, start_date, title="Global Forecast", site="unknown"):
        
        print(f"Original forecast data shapes: X={fc_X.shape}, Y={fc_Y.shape}")
        
        # Ensure X and Y have the same length (they should already)
        min_length = min(len(fc_X), len(fc_Y))
        fc_X = fc_X[:min_length]
        fc_Y = fc_Y[:min_length]
        
        # Adaptive sequence length based on available data
        max_forecast_days = 7
        
        # Check if we have enough data for the full sequence length
        required_length_full = seq_len + max_forecast_days - 1  # Ideal: seq_len + 6 more for 7 forecasts
        
        if min_length < seq_len + 1:  # Need at least seq_len + 1 for one forecast
            print(f"Error: Not enough data for any forecasting. Need at least {seq_len + 1} days, have {min_length}")
            return np.array([]), np.array([]), float('inf')
        
        # Adjust forecast strategy based on available data
        if min_length < required_length_full:
            # Use available data and adjust forecast days
            actual_forecast_days = min_length - seq_len + 1
            print(f"Limited data: Using {seq_len}-day sequences with {actual_forecast_days} forecast days (max {max_forecast_days} requested)")
        else:
            # Use full forecast period
            actual_forecast_days = max_forecast_days
            print(f"Using full sequence length: {seq_len} days with {actual_forecast_days} forecast days")
        
        # Determine how many forecast sequences we can create
        n_forecast_sequences = min_length - seq_len + 1
        actual_forecast_days = min(actual_forecast_days, n_forecast_sequences)
        
        print(f"Can create {n_forecast_sequences} sequences, using last {actual_forecast_days} for forecast")
        print(f"Data shapes for reshape_data: X={fc_X.shape}, Y={fc_Y.shape}")
        
        # Create sequences using available data - use a more flexible approach for forecasting
        if min_length >= seq_len:
            # Create sequences for forecasting - we can predict the last available values
            fc_X_seq = []
            fc_Y_seq = []
            
            # For forecasting, we want to use all available data efficiently
            # Create overlapping sequences that allow us to predict recent values
            for i in range(min_length - seq_len + 1):
                if i + seq_len < min_length:  # Ensure we have a target
                    sequence = fc_X[i:i + seq_len]
                    target = fc_Y[i + seq_len]
                    
                    # Check for NaNs in this sequence
                    if not np.isnan(sequence).any() and not np.isnan(target).any():
                        fc_X_seq.append(sequence)
                        fc_Y_seq.append(target)
            
            if len(fc_X_seq) > 0:
                fc_X_seq = np.array(fc_X_seq)
                fc_Y_seq = np.array(fc_Y_seq)
                if fc_Y_seq.ndim == 2 and fc_Y_seq.shape[1] == 1:
                    fc_Y_seq = fc_Y_seq.squeeze()  # Only squeeze if it's safe
                print(f"After custom sequence creation: X_seq={fc_X_seq.shape}, Y_seq={fc_Y_seq.shape}")
            else:
                print("No valid sequences could be created")
                return np.array([]), np.array([]), float('inf')
        else:
            print(f"Insufficient data for {seq_len}-day sequences")
            return np.array([]), np.array([]), float('inf')
        
        # For forecasting, we only need at least one sequence to make predictions
        # We don't need as many sequences as forecast days
        available_sequences = len(fc_X_seq)
        print(f"Available sequences: {available_sequences}, requested forecast days: {actual_forecast_days}")
        
        # Use the most recent sequence(s) for forecasting
        if available_sequences > 0:
            # Use the last sequence for forecasting
            fc_X_seq = fc_X_seq[-1:]  # Keep only the most recent sequence
            if fc_Y_seq.ndim > 0 and len(fc_Y_seq) > 0:
                fc_Y_seq = fc_Y_seq[-1:]  # Keep corresponding target
            else:
                # Create a dummy target with the right shape for TensorDataset
                fc_Y_seq = np.zeros((1,))  # Single dummy target
            print(f"Using most recent sequence: X_seq={fc_X_seq.shape}, Y_seq={fc_Y_seq.shape}")
        
        print(f"Final shapes for TensorDataset: X_seq={fc_X_seq.shape}, Y_seq={fc_Y_seq.shape}")

        # Create data loader for forecast sequences
        fc_ds = TensorDataset(torch.from_numpy(fc_X_seq).float(), torch.from_numpy(fc_Y_seq).float())
        fc_loader = DataLoader(fc_ds, batch_size=batch_size, shuffle=False)
        self.eval()
        pred_list, obs_list = [], []
        with torch.no_grad():
            for Xb, Yb in fc_loader:
                Xb = Xb.to(device)
                yhat = self(Xb)  # Get normalized predictions
                # Debug: Check for NaNs in model output per batch
                if torch.isnan(yhat).any():
                    print("NaN in model output for this batch!")
                pred_list.append(yhat.cpu().numpy())
                obs_list.append(Yb.numpy())
        
        # Concatenate all batch results (still normalized)
        preds = np.concatenate(pred_list).squeeze()
        obs   = np.concatenate(obs_list).squeeze()
        
        # Calculate RMSE on normalized data
        rmse_fc = np.sqrt(np.mean((preds - obs) ** 2))
        print(f"7-day Forecast RMSE: {rmse_fc:.3f}")
        
        # Calculate persistence baseline using sequence-level approach (like your suggestion)
        persistence_preds_normalized = []
        persistence_rmse_normalized = 0.0
        
        for i in range(len(fc_X_seq)):
            # Get the last temperature value from each sequence (persistence prediction)
            persistent = fc_X_seq[i, -1, -1]  # Last timestep, last feature (temperature)
            target = fc_Y_seq[i]
            
            persistence_preds_normalized.append(persistent)
            # Calculate RMSE on normalized values
            persistence_rmse_normalized += (target - persistent) ** 2
        
        persistence_rmse_normalized = np.sqrt(persistence_rmse_normalized / len(fc_X_seq))
        print(f"Persistence RMSE (normalized): {persistence_rmse_normalized:.3f}")
        
        # Denormalize predictions and observations for interpretable results
        preds_denorm = self.denormalize_predictions(preds)
        obs_denorm = self.denormalize_predictions(obs)
        
        # Ensure preds_denorm is at least 1D for consistent indexing
        if preds_denorm.ndim == 0:
            preds_denorm = np.array([preds_denorm])
        if obs_denorm.ndim == 0:
            obs_denorm = np.array([obs_denorm])
        
        # Limit forecast to 7 days
        max_forecast_days = 7
        if len(preds_denorm) > max_forecast_days:
            print(f"Limiting forecast from {len(preds_denorm)} days to {max_forecast_days} days")
            preds_denorm = preds_denorm[:max_forecast_days]
            obs_denorm = obs_denorm[:max_forecast_days]
            
            # Recalculate RMSE for 7-day period
            rmse_fc = np.sqrt(np.mean((preds[:max_forecast_days] - obs[:max_forecast_days]) ** 2))
            print(f"7-day limited Forecast RMSE: {rmse_fc:.3f}")
        
        # Denormalize persistence predictions
        persistence_preds_denorm = self.denormalize_predictions(np.array(persistence_preds_normalized))
        
        # Ensure 1D array for consistent indexing
        if persistence_preds_denorm.ndim == 0:
            persistence_preds_denorm = np.array([persistence_preds_denorm])
        
        if persistence_preds_denorm.size > max_forecast_days:
            persistence_preds_denorm = persistence_preds_denorm[:max_forecast_days]
        persistence_rmse_denorm = np.sqrt(np.mean((persistence_preds_denorm - obs_denorm) ** 2))
        
        # Create date range for plotting
        forecast_dates = pd.date_range(start=start_date, periods=len(preds_denorm), freq="D")
        # Plot denormalized values with persistence comparison
        plot_results(forecast_dates, obs_denorm, preds_denorm, title=title, rmse=rmse_fc, site=site, 
                    persistence_predictions=persistence_preds_denorm, persistence_rmse=persistence_rmse_denorm)
        return preds_denorm, obs_denorm, rmse_fc

def main():
    # Settings
    DATA_DIR = "data"
    SEQ_LEN = 30
    BATCH_SIZE = 32 
    HIDDEN_SIZE = 32
    N_EPOCHS = 30
    LEARNING_RATE = 0.0001
    NUM_LAYERS = 1    # Number of LSTM layers
    DROPOUT_RATE = 0.0
    
    #-----------DATA PROCESSING--------#
    print("Loading and processing data...")
    (X_train, X_test, 
    Y_train, Y_test, 
    forecast_data_by_site, x_scaler, 
    y_scaler, shape) = global_data_processing.main()

    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Forecast data available for sites: {list(forecast_data_by_site.keys())}")
    
    # Build sequences for LSTM input
    print("Creating sequences...")
    X_train_seq, Y_train_seq = reshape_data(X_train, Y_train, SEQ_LEN)
    X_test_seq,  Y_test_seq  = reshape_data(X_test,  Y_test,  SEQ_LEN)
    
    print(f"Sequences created - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

    # Create PyTorch DataLoaders for batch processing
    train_ds = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(Y_train_seq).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(Y_test_seq).float())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=True)  # Optimized for GPU
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, pin_memory=True)   # Optimized for GPU

    # Initialize model with y_scaler for denormalization, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Training batches per epoch: {len(train_loader)}")
        print(f"Batch size: {BATCH_SIZE}")
        # Enable optimized CUDA operations
        torch.backends.cudnn.benchmark = True
        
    model = GlobalStreamTempLSTM(
        input_size=X_train_seq.shape[2], 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT_RATE,
        y_scaler=y_scaler
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    # Training phase
    train_rmse_history = model.fit(train_loader, optimizer, loss_func, device, n_epochs=N_EPOCHS, test_loader=test_loader)

    # Evaluation phase with denormalized results
    date_range = pd.date_range(start='2021-04-16', periods=len(Y_test_seq), freq='D')
    preds_denorm, obs_denorm, rmse = model.evaluate(test_loader, device, date_range=date_range, title="Global Model", site="global")

    # Start 7-day forecast from day after test period ends for each site
    daily_rmse_comparisons = {}  # Store RMSE comparisons for all sites
    site_rmse_dict = {}  # Store RMSE for each site
    
    for site in SITE_IDS:
        print(f"\n=== Processing 7-day forecast for site {site} ===")
        
        # Get forecast data for this site
        if site in forecast_data_by_site and len(forecast_data_by_site[site]['X']) > 0:
            site_forecast_X = forecast_data_by_site[site]['X']
            site_forecast_Y = forecast_data_by_site[site]['Y']
            
            print(f"Site {site} forecast data shape: X={site_forecast_X.shape}, Y={site_forecast_Y.shape}")
            
            forecast_start_date = date_range[-1] + pd.Timedelta(days=1)
            
            fc_preds_denorm, fc_obs_denorm, fc_rmse = model.forecast(site_forecast_X, site_forecast_Y, SEQ_LEN, BATCH_SIZE, device, 
                                                                start_date=forecast_start_date, title=f"Site {site} Forecast", site=site)
            
            # Only process results if we got valid forecasts
            if len(fc_preds_denorm) > 0 and not np.isinf(fc_rmse):
                print(f"Forecast completed for site: {site} - RMSE: {fc_rmse:.3f}")
                site_rmse_dict[site] = fc_rmse
                
                # Calculate daily RMSE for 7-day forecast comparison
                daily_rmse_comparison = calculate_daily_forecast_rmse(fc_preds_denorm, fc_obs_denorm, forecast_start_date)
                daily_rmse_comparisons[site] = daily_rmse_comparison
            else:
                print(f"Forecast failed for site: {site} - insufficient data")
                site_rmse_dict[site] = float('inf')
        else:
            print(f"Warning: No forecast data available for site {site}")
    
    # Save results summary
    if train_rmse_history and site_rmse_dict:
        save_results_summary(train_rmse_history, rmse, site_rmse_dict)
    
    # Plot 7-day RMSE comparison for the last processed site (or you could plot all sites)
    if daily_rmse_comparisons:
        # Find a site with valid data
        valid_sites = [site for site, data in daily_rmse_comparisons.items() 
                      if len(data.get('lstm_rmse', [])) > 0]
        
        if valid_sites:
            last_site = valid_sites[-1]
            print(f"\nPlotting 7-day RMSE comparison for site {last_site}")
            plot_7day_rmse_comparison(daily_rmse_comparisons[last_site])
        else:
            print("\nNo valid forecast data available for RMSE comparison plotting")
    else:
        print("\nNo daily RMSE comparisons available for plotting")

def calculate_daily_forecast_rmse(predictions, observations, start_date):
    """Calculate RMSE for each day of the forecast period and persistence baseline"""
    # This function now receives denormalized predictions and observations
    # For the 7-day comparison, we'll calculate day-by-day RMSE from the sequence predictions
    
    # Group predictions and observations by day
    forecast_dates = pd.date_range(start=start_date, periods=len(predictions), freq="D")
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'date': forecast_dates,
        'prediction': predictions,
        'observation': observations
    })
    
    # Calculate daily RMSE for LSTM model
    daily_lstm_rmse = []
    daily_persistence_rmse = []
    
    # For persistence baseline: use a proper sequence-based approach
    # Since we don't have access to the original sequences here, we'll use a simplified day-by-day approach
    # This is less accurate than the sequence-level approach used in the main forecast method
    
    for i in range(min(7, len(df))):  # Limit to 7 days
        day_data = df.iloc[i:i+1]
        
        # LSTM RMSE for this day
        lstm_rmse = np.sqrt(np.mean((day_data['prediction'] - day_data['observation']) ** 2))
        daily_lstm_rmse.append(lstm_rmse)
        
        # Simplified persistence RMSE (this is less accurate than sequence-based approach)
        if i == 0:
            # For the first forecast day, use mean as baseline
            persistence_pred = np.mean(observations)
        else:
            # For subsequent days, use the previous day's actual observation
            persistence_pred = observations[i-1]
            
        actual_obs = day_data['observation'].iloc[0]
        persistence_rmse = abs(persistence_pred - actual_obs)  # Use absolute error for daily comparison
        daily_persistence_rmse.append(persistence_rmse)
    
    return {
        'days': list(range(1, len(daily_lstm_rmse) + 1)),
        'lstm_rmse': daily_lstm_rmse,
        'persistence_rmse': daily_persistence_rmse
    }

def save_results_summary(model_rmse, test_rmse, site_rmse_dict):
    """Save a summary of all results to a text file"""
    
    filename = f"results_summary.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write("=== LSTM Stream Temperature Forecasting Results ===\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write(f"Final Training RMSE: {model_rmse[-1]:.3f}\n")
        f.write(f"Test RMSE: {test_rmse:.3f}\n")
        f.write(f"Training Epochs: {len(model_rmse)}\n\n")
        
        f.write("SITE-SPECIFIC FORECAST RESULTS:\n")
        for site, rmse in site_rmse_dict.items():
            f.write(f"Site {site}: {rmse:.3f} RMSE\n")
        
        f.write(f"\nAverage Site RMSE: {np.mean(list(site_rmse_dict.values())):.3f}\n")
        f.write(f"Best Site RMSE: {min(site_rmse_dict.values()):.3f} (Site {min(site_rmse_dict, key=site_rmse_dict.get)})\n")
        f.write(f"Worst Site RMSE: {max(site_rmse_dict.values()):.3f} (Site {max(site_rmse_dict, key=site_rmse_dict.get)})\n")
        
        f.write("\nEPOCH-BY-EPOCH TRAINING RMSE:\n")
        for i, rmse in enumerate(model_rmse, 1):
            f.write(f"Epoch {i:2d}: {rmse:.3f}\n")
    
    print(f"Results summary saved: {filepath}")
    return filepath

def plot_7day_rmse_comparison(daily_rmse_data):
    """Plot RMSE comparison over 7 days between LSTM and persistence"""
    
    # Check if we have valid data
    if not daily_rmse_data or len(daily_rmse_data.get('lstm_rmse', [])) == 0:
        print("Warning: No valid RMSE data available for plotting")
        return
    
    plt.figure(figsize=(12, 6))
    
    days = daily_rmse_data['days']
    lstm_rmse = daily_rmse_data['lstm_rmse']
    persistence_rmse = daily_rmse_data['persistence_rmse']
    
    # Check if we have enough data points
    if len(days) == 0:
        print("Warning: No forecast days available for plotting")
        return
    
    # Create line plot
    plt.plot(days, lstm_rmse, 'b-', label='LSTM Forecast', linewidth=2.5, marker='o', markersize=6)
    plt.plot(days, persistence_rmse, 'r--', label='Persistence Baseline', linewidth=2.5, marker='s', markersize=6)
    
    plt.xlabel('Forecast Day', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Daily RMSE Comparison: LSTM vs Persistence (7-Day Forecast)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all days (avoid singular limits)
    plt.xticks(days)
    if len(days) > 1:
        plt.xlim(min(days) - 0.5, max(days) + 0.5)
    else:
        plt.xlim(0.5, 1.5)  # Single day case
    
    # Add average improvement (only if we have valid data)
    if len(lstm_rmse) > 0 and len(persistence_rmse) > 0:
        avg_lstm = np.mean(lstm_rmse)
        avg_persistence = np.mean(persistence_rmse)
        
        if avg_persistence > 0:  # Avoid division by zero
            avg_improvement = ((avg_persistence - avg_lstm) / avg_persistence) * 100
        else:
            avg_improvement = 0.0
    else:
        avg_lstm = avg_persistence = avg_improvement = 0.0
    
    # Add value annotations for key points
    for i, (day, lstm_val, pers_val) in enumerate(zip(days, lstm_rmse, persistence_rmse)):
        if i % 2 == 0:  # Annotate every other day to avoid clutter
            plt.annotate(f'{lstm_val:.3f}', (day, lstm_val), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color='blue')
            plt.annotate(f'{pers_val:.3f}', (day, pers_val), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Add improvement text
    improvement_text = f'''Average Performance:
• LSTM: {avg_lstm:.3f} RMSE
• Persistence: {avg_persistence:.3f} RMSE
• Improvement: {avg_improvement:.1f}%'''
    
    plt.text(0.02, 0.98, improvement_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"7day_rmse_comparison.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"7-day RMSE comparison plot saved: {filepath}")
    
    #plt.show()

if __name__ == "__main__":
    main()
    