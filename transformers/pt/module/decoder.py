import math
import torch.nn as nn

from module.postional_encoding import PositionalEncoder
from module.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout) -> None:
        super().__init__()
        self.d_model =d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoder(max_len, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, hidden, dropout) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_output, dec_input, padding_mask, look_ahead_mask):
        embedding = self.embedding(dec_input)
        embedding *= math.sqrt(self.d_model)
        embedding = self.pos_embedding(embedding)

        outputs = self.dropout(embedding)

        for layer in self.dec_layers:
            outputs = layer(outputs, enc_output, padding_mask, look_ahead_mask)

        return outputs