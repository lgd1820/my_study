import torch.nn as nn

from module.multihead_attention import MultiHeadAttention
from module.positionwise_feed_forward import PositionwiseFeedForwad

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden, dropout) -> None:
        super().__init__()

        self.attention1 = MultiHeadAttention(d_model, n_heads)
        self.attention2 = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForwad(d_model, hidden)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs, enc_outputs, padding_mask, look_ahead_mask):
        attention1 = self.attention1({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
        attention1 = self.norm1(inputs + attention1)

        attention2 = self.attention2({'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})
        attention2 = self.dropout1(attention2)
        attention2 = self.norm2(attention1 + attention2)

        outputs = self.ff(attention2)
        outputs = self.dropout3(outputs)
        outputs = self.norm3(attention2 + outputs)

        return outputs