import torch.nn as nn

from layers.multi_head_attention import MultiHeadAttention
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob) -> None:
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = 