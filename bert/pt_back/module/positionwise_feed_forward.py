import torch
import torch.nn as nn

class PositionwiseFeedForwad(nn.Module):
    def __init__(self, d_model, hidden) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(d_model, hidden)
        self.fc_2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        outputs = self.fc_1(x)
        outputs = self.relu(outputs)
        outputs = self.fc_2(outputs)

        return outputs