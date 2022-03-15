import torch
import torch.nn as nn
import math 

class PositionalEncoding(nn.Module):
    """
    위치 임베딩
    """
    def __init__(self, d_model, max_len, device) -> None:
        """
            d_model: 모델의 차원
            max_len: 시퀀스의 최대 길이
            device: 하드웨어 디바이스 세팅
        """
        super(PositionalEncoding, self).__init__()

        # input size와 같은 크기        
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        # i는 d_model의 인덱스
        # step은 i의 멀티플 2 * i
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos/10000 ** (_2i / d_model))
        self.encoding[:, 1::2] = torch.cos(pos/10000 ** (_2i / d_model))

    def forward(self, x):
        batch_size, seq_size = x.size()

        return self.encoding[:seq_size, :]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out
        
class ScaleDotProductAttention(nn.Module):
    """
    스케일 갓 프로덕트 어텐션
    쿼리, 키, 밸류
    """
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
    
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)
        
        score = self.softmax(score)

        v = score @ v

        return v, score