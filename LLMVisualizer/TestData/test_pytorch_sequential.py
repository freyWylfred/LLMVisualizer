import torch
import torch.nn as nn
from collections import OrderedDict

class SequentialModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dim, hidden_dim)),
            ('relu1', nn.ReLU()),
            ('bn1', nn.BatchNorm1d(hidden_dim)),
            ('fc2', nn.Linear(hidden_dim, hidden_dim)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc3', nn.Linear(hidden_dim, output_dim)),
        ]))
        self.branches = nn.ModuleDict({
            'classifier': nn.Linear(output_dim, 5),
            'regressor': nn.Linear(output_dim, 1),
        })

    def forward(self, x):
        x = self.features(x)
        cls_out = self.branches['classifier'](x)
        reg_out = self.branches['regressor'](x)
        return cls_out, reg_out
