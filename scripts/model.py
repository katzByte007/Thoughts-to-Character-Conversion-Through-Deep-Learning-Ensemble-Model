import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 50, 1)
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(self.relu(out))
        return out

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes, kernel_size=3, dropout=0.2):
        super(CNNLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(16, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 50, 1)  # Add a channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x