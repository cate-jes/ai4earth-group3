import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import global_data_processing

SITE_IDS = [1450, 1565, 1571, 1573, 1641]

seed = 42
torch.manual_seed(seed)
    # If using CUDA, also set the CUDA seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) # for a single GPU
    # or torch.cuda.manual_seed_all(seed) # for all GPUs

# Utility: reshape data for LSTM
def reshape_data(x, y, seq_length):
    n_samples, n_features = x.shape
    n_sequences = n_samples - seq_length + 1
    x_seq = np.zeros((n_sequences, seq_length, n_features), dtype=x.dtype)
    y_seq = np.zeros((n_sequences, 1), dtype=y.dtype)
    for i in range(n_sequences):
        x_seq[i] = x[i : i + seq_length]
        y_seq[i] = y[i + seq_length - 1]
    return x_seq, y_seq

# Plotting function
def plot_results(date_range, observations, predictions, title, rmse, site):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(date_range, observations, label="Observation", linewidth=1, linestyle='-', marker='o', markersize=2)
    ax.plot(date_range, predictions, label="Prediction", linewidth=1, linestyle='--', marker='x', markersize=2)
    ax.legend(fontsize='large')
    ax.set_title(f"{title} - Test set RMSE: {rmse:.3f}", fontsize=16)
    ax.set_xlabel("", fontsize=14)
    ax.set_ylabel("Temperature (°C)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    plt.show()

class GlobalStreamTempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, y_scaler=None):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # Store the y_scaler for denormalization of predictions
        self.y_scaler = y_scaler
        
    def denormalize_predictions(self, normalized_data):
        """Convert normalized predictions/targets back to original temperature scale"""
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(normalized_data.reshape(-1, 1)).squeeze()
        return normalized_data
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(self.dropout(last_hidden))
        return out

    def fit(self, train_loader, optimizer, loss_func, device, n_epochs=30, test_loader=None):
        # Add GPU memory monitoring and timing
        import time
        self.train()
        
        # Store RMSE values for plotting
        model_rmse_history = []
        persistence_rmse_history = []
        
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
            epoch_losses = []
            
            # Clear GPU cache before each epoch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
            for batch_idx, (Xb, Yb) in enumerate(train_loader):
                
                # Move data to GPU
                Xb, Yb = Xb.to(device), Yb.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                preds = self(Xb)
                
                # Check for NaN in predictions
                if torch.isnan(preds).any():
                    print(f"NaN in predictions at epoch {epoch}, batch {batch_idx}")
                    print(f"Input min: {Xb.min():.6f}, max: {Xb.max():.6f}")
                    print(f"Target min: {Yb.min():.6f}, max: {Yb.max():.6f}")
                    # Skip this batch
                    continue
                
                loss = loss_func(preds, Yb)
                
                # Check for NaN in loss
                if torch.isnan(loss).any():
                    print(f"NaN in loss at epoch {epoch}, batch {batch_idx}")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(loss.item())

            # Calculate model RMSE for this epoch
            if epoch_losses:  # Only if we have valid losses
                avg_loss = np.mean(epoch_losses)
                model_rmse = np.sqrt(avg_loss)
                model_rmse_history.append(model_rmse)
                
                # Calculate persistence RMSE if test_loader is provided
                if test_loader is not None:
                    persistence_rmse = self._calculate_persistence_rmse(test_loader, device)
                    persistence_rmse_history.append(persistence_rmse)
                    print(f"Epoch {epoch} done. - Model RMSE: {model_rmse:.3f}, Persistence RMSE: {persistence_rmse:.3f}")
                else:
                    print(f"Epoch {epoch} done. - Model RMSE: {model_rmse:.3f}")
            else:
                print(f"Epoch {epoch} failed - no valid losses computed")
        
        # Plot RMSE comparison if we have both model and persistence data
        if test_loader is not None and model_rmse_history:
            self._plot_rmse_comparison(model_rmse_history, persistence_rmse_history)
    
    def _calculate_persistence_rmse(self, test_loader, device):
        """Calculate RMSE for persistence baseline (using previous day's temperature)"""
        self.eval()
        persistence_errors = []
        
        with torch.no_grad():
            for Xb, Yb in test_loader:
                # For persistence, use the last temperature value from the sequence
                # as the prediction for the next day
                persistence_pred = Xb[:, -1, 0]  # Assuming temperature is the first feature
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
        plt.show()
           
    def evaluate(self, test_loader, device, date_range=None, title="Global Model", site="global"):
        self.eval()
        preds, obs = [], []
        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(device)
                pred = self(Xb).cpu().numpy()  
                preds.append(pred)
                obs.append(Yb.numpy())  
        
        # Concatenate all batch results
        preds = np.concatenate(preds).squeeze()
        obs = np.concatenate(obs).squeeze()
        
        # Calculate RMSE 
        rmse = np.sqrt(np.mean((preds - obs) ** 2))
        print(f"Test RMSE: {rmse:.3f}")
        
        preds_denorm = self.denormalize_predictions(preds)
        obs_denorm = self.denormalize_predictions(obs)
        
        if date_range is not None:
            # Plot denormalized values for meaningful visualization
            plot_results(date_range, obs_denorm, preds_denorm, title=title, rmse=rmse, site=site)
        return preds_denorm, obs_denorm, rmse

    def forecast(self, fc_X, fc_Y, seq_len, batch_size, device, start_date, title="Global Forecast", site="unknown"):
        # Debug: Check for NaNs/Infs in forecast input data
        print("Any NaN in fc_X?", np.isnan(fc_X).any())
        print("Any Inf in fc_X?", np.isinf(fc_X).any())
        print("Any NaN in fc_Y?", np.isnan(fc_Y).any())
        print("Any Inf in fc_Y?", np.isinf(fc_Y).any())
        print("fc_X shape:", fc_X.shape)
        print("Model input size:", self.lstm.input_size)
        
        fc_X_seq, fc_Y_seq = reshape_data(fc_X, fc_Y, seq_len)
        # Debug: Check for NaNs/Infs after sequence creation
        print("Any NaN in fc_X_seq?", np.isnan(fc_X_seq).any())
        print("Any Inf in fc_X_seq?", np.isinf(fc_X_seq).any())
        print("fc_X_seq min:", np.nanmin(fc_X_seq), "max:", np.nanmax(fc_X_seq))
        print("fc_Y_seq min:", np.nanmin(fc_Y_seq), "max:", np.nanmax(fc_Y_seq))
        
        # Remove rows with NaNs - entire sequences are removed if any value is missing
        mask = ~np.isnan(fc_X_seq).any(axis=(1,2)) & ~np.isnan(fc_Y_seq).any(axis=1)
        X_train = fc_X_seq[mask]  # Keep only clean sequences
        Y_train = fc_Y_seq[mask].squeeze()  # Remove singleton dimensions
        
        fc_X_seq= X_train
        fc_Y_seq = Y_train

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
        print("Any NaN in preds?", np.isnan(preds).any())
        print("Any NaN in obs?", np.isnan(obs).any())
        print("preds shape:", preds.shape, "obs shape:", obs.shape)
        
        # Calculate RMSE on normalized data
        rmse_fc = np.sqrt(np.mean((preds - obs) ** 2))
        print(f"7-day Forecast RMSE: {rmse_fc:.3f}")
        
        # Denormalize predictions and observations for interpretable results
        preds_denorm = self.denormalize_predictions(preds)
        obs_denorm = self.denormalize_predictions(obs)
        
        # Create date range for plotting
        forecast_dates = pd.date_range(start=start_date, periods=len(preds_denorm), freq="D")
        # Plot denormalized values for meaningful visualization
        plot_results(forecast_dates, obs_denorm, preds_denorm, title=title, rmse=rmse_fc, site=site)
        return preds_denorm, obs_denorm, rmse_fc

def main():
    DATA_DIR = "data"
    SEQ_LEN = 30
    BATCH_SIZE = 64  # Increased batch size for better GPU utilization
    HIDDEN_SIZE = 64  # Increased model size to better utilize GPU
    N_EPOCHS = 30
    LEARNING_RATE = 0.0001
    NUM_LAYERS = 1  # Number of LSTM layers
    
    #-----------DATA PROCESSING--------#
    print("Loading and processing data...")
    (X_train, X_test, 
    Y_train, Y_test, 
    forecast_data_by_site, x_scaler, 
    y_scaler, shape) = global_data_processing.main()

    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Forecast data available for sites: {list(forecast_data_by_site.keys())}")
    
    # Debug: Check data for NaN/Inf values
    print("Checking training data for NaN/Inf...")
    print(f"X_train - NaN: {np.isnan(X_train).any()}, Inf: {np.isinf(X_train).any()}")
    print(f"X_train min: {np.min(X_train):.6f}, max: {np.max(X_train):.6f}")
    print(f"Y_train - NaN: {np.isnan(Y_train).any()}, Inf: {np.isinf(Y_train).any()}")
    print(f"Y_train min: {np.min(Y_train):.6f}, max: {np.max(Y_train):.6f}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_train first 5 values: {Y_train[:5].flatten()}")
    
    # Build sequences for LSTM input
    print("Creating sequences...")
    X_train_seq, Y_train_seq = reshape_data(X_train, Y_train, SEQ_LEN)
    X_test_seq,  Y_test_seq  = reshape_data(X_test,  Y_test,  SEQ_LEN)
    
    print(f"Sequences created - Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
    
    # Debug: Check sequence data
    print("Checking sequence data for NaN/Inf...")
    print(f"X_train_seq - NaN: {np.isnan(X_train_seq).any()}, Inf: {np.isinf(X_train_seq).any()}")
    print(f"Y_train_seq - NaN: {np.isnan(Y_train_seq).any()}, Inf: {np.isinf(Y_train_seq).any()}")
    print(f"Y_train_seq min: {np.min(Y_train_seq):.6f}, max: {np.max(Y_train_seq):.6f}")
    print(f"Y_train_seq shape: {Y_train_seq.shape}")
    print(f"Y_train_seq first 5 values: {Y_train_seq[:5].flatten()}")

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
        
    model = GlobalStreamTempLSTM(input_size=X_train_seq.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, y_scaler=y_scaler).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    # Training phase
    model.fit(train_loader, optimizer, loss_func, device, n_epochs=N_EPOCHS, test_loader=test_loader)

    # Evaluation phase with denormalized results
    date_range = pd.date_range(start='2021-04-16', periods=len(Y_test_seq), freq='D')
    preds_denorm, obs_denorm, rmse = model.evaluate(test_loader, device, date_range=date_range, title="Global Model", site="global")

    # Start forecast from day after test period ends for each site
    daily_rmse_comparisons = {}  # Store RMSE comparisons for all sites
    
    for site in SITE_IDS:
        print(f"\n=== Processing forecast for site {site} ===")
        
        # Get forecast data for this site
        if site in forecast_data_by_site and len(forecast_data_by_site[site]['X']) > 0:
            site_forecast_X = forecast_data_by_site[site]['X']
            site_forecast_Y = forecast_data_by_site[site]['Y']
            
            print(f"Site {site} forecast data shape: X={site_forecast_X.shape}, Y={site_forecast_Y.shape}")
            
            forecast_start_date = date_range[-1] + pd.Timedelta(days=1)
            fc_preds_denorm, fc_obs_denorm, fc_rmse = model.forecast(site_forecast_X, site_forecast_Y, SEQ_LEN, BATCH_SIZE, device, 
                                                                start_date=forecast_start_date, title=f"Site {site} Forecast", site=site)
            print(f"Forecast completed for site: {site} - RMSE: {fc_rmse:.3f}")

            # Calculate daily RMSE for 7-day forecast comparison
            daily_rmse_comparison = calculate_daily_forecast_rmse(fc_preds_denorm, fc_obs_denorm, forecast_start_date)
            daily_rmse_comparisons[site] = daily_rmse_comparison
        else:
            print(f"Warning: No forecast data available for site {site}")
    
    # Plot 7-day RMSE comparison for the last processed site (or you could plot all sites)
    if daily_rmse_comparisons:
        last_site = list(daily_rmse_comparisons.keys())[-1]
        print(f"\nPlotting 7-day RMSE comparison for site {last_site}")
        plot_7day_rmse_comparison(daily_rmse_comparisons[last_site])

def calculate_daily_forecast_rmse(predictions, observations, start_date):
    """Calculate RMSE for each day of the forecast period and persistence baseline"""
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
    
    # For persistence baseline, we'll use the observation from the previous day
    # For the first day, we'll use the last known temperature from training data
    prev_temp = observations[0]  # Use first observation as baseline
    
    for i in range(min(7, len(df))):  # Limit to 7 days
        day_data = df.iloc[i:i+1]
        
        # LSTM RMSE for this day
        lstm_rmse = np.sqrt(np.mean((day_data['prediction'] - day_data['observation']) ** 2))
        daily_lstm_rmse.append(lstm_rmse)
        
        # Persistence RMSE for this day (using previous day's temperature)
        persistence_pred = prev_temp
        persistence_rmse = np.sqrt(np.mean((persistence_pred - day_data['observation']) ** 2))
        daily_persistence_rmse.append(persistence_rmse)
        
        # Update previous temperature for next iteration
        prev_temp = day_data['observation'].iloc[0]
    
    return {
        'days': list(range(1, len(daily_lstm_rmse) + 1)),
        'lstm_rmse': daily_lstm_rmse,
        'persistence_rmse': daily_persistence_rmse
    }

def plot_7day_rmse_comparison(daily_rmse_data):
    """Plot RMSE comparison over 7 days between LSTM and persistence"""
    plt.figure(figsize=(12, 6))
    
    days = daily_rmse_data['days']
    lstm_rmse = daily_rmse_data['lstm_rmse']
    persistence_rmse = daily_rmse_data['persistence_rmse']
    
    # Create line plot
    plt.plot(days, lstm_rmse, 'b-', label='LSTM Forecast', linewidth=2.5, marker='o', markersize=6)
    plt.plot(days, persistence_rmse, 'r--', label='Persistence Baseline', linewidth=2.5, marker='s', markersize=6)
    
    plt.xlabel('Forecast Day', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Daily RMSE Comparison: LSTM vs Persistence (7-Day Forecast)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show all 7 days
    plt.xticks(days)
    plt.xlim(0.5, len(days) + 0.5)
    
    # Add average improvement
    avg_lstm = np.mean(lstm_rmse)
    avg_persistence = np.mean(persistence_rmse)
    avg_improvement = ((avg_persistence - avg_lstm) / avg_persistence) * 100
    
    # Add value annotations for key points
    for i, (day, lstm_val, pers_val) in enumerate(zip(days, lstm_rmse, persistence_rmse)):
        if i % 2 == 0:  # Annotate every other day to avoid clutter
            plt.annotate(f'{lstm_val:.3f}', (day, lstm_val), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color='blue')
            plt.annotate(f'{pers_val:.3f}', (day, pers_val), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=9, color='red')
    
    # Add improvement 
    improvement_text = f'''Average Performance:
• LSTM: {avg_lstm:.3f} RMSE
• Persistence: {avg_persistence:.3f} RMSE
• Improvement: {avg_improvement:.1f}%'''
    
    plt.text(0.02, 0.98, improvement_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    