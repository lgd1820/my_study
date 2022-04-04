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
        """
        Args:
            inputs : 입력 데이터
            enc_outputs : 인코더의 표현값(R)
            padding_mask : 패딩 마스크
            look_ahead_mask : 마스크

        Returns:
            outputs : 디코더의 표현값
        """

        # 마스크드 멀티 헤드 어텐션 (M)
        attention1 = self.attention1({'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})
        attention1 = self.norm1(inputs + attention1)

        # 멀티 헤드 어텐션 (R, M) : 인코더-디코더 어텐션 레이어
        attention2 = self.attention2({'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})

        # 드롭아웃 + 잔차연결및 층 정규화
        attention2 = self.dropout1(attention2)
        attention2 = self.norm2(attention1 + attention2)

        # 피드포워드
        outputs = self.ff(attention2)

        # 드롭아웃 + 잔차연결및 층 정규화
        outputs = self.dropout3(outputs)
        outputs = self.norm3(attention2 + outputs)

        return outputs