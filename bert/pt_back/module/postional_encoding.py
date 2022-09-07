import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, position, d_model) -> None:
        super().__init__()
        self.d_model = d_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos_encoding = torch.zeros(position, d_model).to(device)

        pos = torch.arange(0, position).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2).float()

        pos_encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i / d_model))
        pos_encoding[:, 1::2] = torch.cos(pos / 10000 ** (_2i / d_model))
        self.pos_encoding = pos_encoding.unsqueeze(0)
        self.pos_encoding.requires_grad = False
    
    def forward(self, x):
        return math.sqrt(self.d_model) * x + self.pos_encoding[:, :x.size(1)]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_embedding = PositionalEncoder(60, 256)
    random = torch.rand(64, 60, 256).to(device)

    result = pos_embedding(random)
    print(result.shape)