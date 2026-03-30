import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 512
nhead = 8
num_layers = 6

class AdvancedTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=32000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.dropout(src)
        memory = self.encoder(src)
        tgt = self.embedding(tgt)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output)
        return output
