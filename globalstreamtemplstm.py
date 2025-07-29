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
def plot_results(date_range, observations, predictions, title, rmse):
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

# LSTM Model
class GlobalStreamTempLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, y_scaler=None):
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

    def fit(self, train_loader, optimizer, loss_func, device, n_epochs=30):
        for epoch in range(1, n_epochs+1):
            self.train()
            for Xb, Yb in train_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                optimizer.zero_grad()
                preds = self(Xb)
                loss = loss_func(preds, Yb)
                loss.backward()
                optimizer.step()
            rmse_value = np.sqrt(loss.item())
            print(f"Epoch {epoch} done. - Test RMSE: {rmse_value:.3f}")

    def evaluate(self, test_loader, device, date_range=None, title="Global Model"):
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
            plot_results(date_range, obs_denorm, preds_denorm, title=title, rmse=rmse)
        return preds_denorm, obs_denorm, rmse

    def forecast(self, fc_X, fc_Y, seq_len, batch_size, device, start_date, title="Global Forecast"):
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
        plot_results(forecast_dates, obs_denorm, preds_denorm, title=title, rmse=rmse_fc)
        return preds_denorm, obs_denorm, rmse_fc

def main():
    DATA_DIR = "data"
    SEQ_LEN = 30
    BATCH_SIZE = 64
    HIDDEN_SIZE = 64
    N_EPOCHS = 30
    LEARNING_RATE = 0.001
    
    (X_train, X_test, 
    Y_train, Y_test, 
    fc_X_df, fc_Y_df, x_scaler, 
    y_scaler, shape) = global_data_processing.main()

    # # Load global data (already normalized from global_data_processing.py)
    # X_train = pd.read_csv(f"{DATA_DIR}\\finetune_global_X_train.csv").values
    # Y_train = pd.read_csv(f"{DATA_DIR}\\finetune_global_Y_train.csv").values.squeeze()
    # X_test  = pd.read_csv(f"{DATA_DIR}\\finetune_global_input_X_test.csv").values
    # Y_test  = pd.read_csv(f"{DATA_DIR}\\finetune_global_Y_test.csv").values.squeeze()

    # # Create y_scaler to match the normalization used in global_data_processing.py
    # # We need to fit it on the original (pre-normalized) training data to get correct parameters
    # # Since we don't have access to original data, we'll create a scaler from normalized data
    # # This is a workaround - ideally the scaler should be saved from preprocessing
    # y_scaler = StandardScaler()
    # y_scaler.fit(Y_train.reshape(-1, 1))  # Fit on normalized training targets
    
    # Build sequences for LSTM input
    X_train_seq, Y_train_seq = reshape_data(X_train, Y_train, SEQ_LEN)
    X_test_seq,  Y_test_seq  = reshape_data(X_test,  Y_test,  SEQ_LEN)

    # Create PyTorch DataLoaders for batch processing
    train_ds = TensorDataset(torch.from_numpy(X_train_seq).float(), torch.from_numpy(Y_train_seq).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test_seq).float(), torch.from_numpy(Y_test_seq).float())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model with y_scaler for denormalization, optimizer, and loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GlobalStreamTempLSTM(input_size=X_train_seq.shape[2], hidden_size=HIDDEN_SIZE, y_scaler=y_scaler).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    # Training phase
    model.fit(train_loader, optimizer, loss_func, device, n_epochs=N_EPOCHS)

    # Evaluation phase with denormalized results
    date_range = pd.date_range(start='2021-04-16', periods=len(Y_test_seq), freq='D')
    preds_denorm, obs_denorm, rmse = model.evaluate(test_loader, device, date_range=date_range, title="Global Model")
    
    # # Forecasting phase
    # fc_X_df = pd.read_csv(f"{DATA_DIR}\\forecast_global_input_X.csv")
    # fc_Y_df = pd.read_csv(f"{DATA_DIR}\\forecast_global_input_Y.csv")
    # fc_X = fc_X_df.values
    # fc_Y = fc_Y_df.values.squeeze()
    
    # Start forecast from day after test period ends
    forecast_start_date = date_range[-1] + pd.Timedelta(days=1)
    fc_preds_denorm, fc_obs_denorm, fc_rmse = model.forecast(fc_X_df, fc_Y_df, SEQ_LEN, BATCH_SIZE, device, 
                                                           start_date=forecast_start_date, title="Global Forecast")

if __name__ == "__main__":
    main()
    