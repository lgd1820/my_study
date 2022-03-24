import torch.nn as nn

from module.multihead_attention import MultiHeadAttention
from module.positionwise_feed_forward import PositionwiseFeedForwad

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden, dropout) -> None:
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForwad(d_model, hidden)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        # 멀티 헤드 어텐션
        attention = self.attention(
            {
                "query" : inputs, 
                "key" : inputs,
                "value" : inputs,
                "mask" : mask
            }
        )

        # 드롭아웃 + 잔차연결및 층 정규화
        attention = self.dropout1(attention)
        attention = self.norm1(inputs + attention)

        # 포지션 와이즈 피드 포워드
        outputs = self.ff(attention)

        # 드롭아웃 + 잔차연결및 층 정규화
        outputs = self.dropout2(outputs)
        outputs = self.norm2(attention + outputs)

        return outputs