import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

# 1. Load preprocessed CSV data
def load_data(data_folder):
    X_train = pd.read_csv(f"{data_folder}/pretrain_site_input_X_train.csv").values
    Y_train = pd.read_csv(f"{data_folder}/pretrain_site_input_Y_train.csv").values
    X_test  = pd.read_csv(f"{data_folder}/pretrain_site_input_X_test.csv").values
    Y_test  = pd.read_csv(f"{data_folder}/pretrain_site_input_Y_test.csv").values
    return X_train, Y_train, X_test, Y_test

    # 2. Create sliding-window sequences for LSTM
def create_sequences(X, Y, seq_len):
    sequences, targets = [], []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
        targets.append(Y[i+seq_len])
    return np.stack(sequences), np.stack(targets)

# 3. Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# 4. Training loop
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device).float()
        Y_batch = Y_batch.to(device).float()
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', '-d', default='data',
                        help='Folder containing preprocessed CSVs')
    parser.add_argument('--seq-len',    '-l', type=int, default=10,
                        help='Sequence length for LSTM')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden-size','-H', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--epochs',     '-e', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr',         '-r', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    # Load data
    X_train, Y_train, X_test, Y_test = load_data('/content/')

    # Build sequences
    X_seq_train, Y_seq_train = create_sequences(X_train, Y_train, seq_len=args.seq_len)
    X_seq_test,  Y_seq_test  = create_sequences(X_test,  Y_test,  seq_len=args.seq_len)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_seq_train), torch.from_numpy(Y_seq_train))
    test_ds  = TensorDataset(torch.from_numpy(X_seq_test),  torch.from_numpy(Y_seq_test))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=X_train.shape[1], hidden_size=args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'lstm_model.pth')