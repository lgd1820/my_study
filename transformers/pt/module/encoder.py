import math
import torch.nn as nn

from module.encoder_layer import EncoderLayer
from module.postional_encoding import PositionalEncoder

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout) -> None:
        super().__init__()
        self.d_model =d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoder(max_len, d_model)
        self.enc_layers = nn.ModuleList([ EncoderLayer(d_model, n_heads, hidden, dropout) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        embedding = self.embedding(x)
        embedding *= math.sqrt(self.d_model)
        embedding = self.pos_embedding(embedding)

        outputs = self.dropout(embedding)

        for layer in self.enc_layers:
            outputs = layer(outputs, mask)
        
        return outputs