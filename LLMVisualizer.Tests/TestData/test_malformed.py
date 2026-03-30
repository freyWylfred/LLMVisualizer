# Edge case: malformed Python with unbalanced parentheses
import torch.nn as nn

class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20
        # Missing closing paren above
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
