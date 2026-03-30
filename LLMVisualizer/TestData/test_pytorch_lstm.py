import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_layers=2, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.fc(x)
        return x
