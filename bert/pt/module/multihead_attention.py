import torch
import torch.nn as nn

from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> torch.Tensor:
        super().__init__()

        assert d_model & n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        q, k, v, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = q.size(0)

        # WQ, WK, WV
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        # 헤드 나누기
        q = rearrange(q, "bs seq_len (n_heads head_dim) -> bs n_heads seq_len head_dim", n_heads=self.n_heads)
        k_T = rearrange(k, "bs seq_len (n_heads head_dim) -> bs n_heads head_dim seq_len", n_heads=self.n_heads)
        v = rearrange(v, "bs seq_len (n_heads head_dim) -> bs n_heads seq_len head_dim", n_heads=self.n_heads)

        score = torch.matmul(q, k_T)

        # MASK
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e-12)
        
        # SCALED DOT PRODUCT
        score = torch.softmax(score, dim=-1)
        result = torch.matmul(score, v)

        # CONCATE
        result = rearrange(result, "bs n_heads seq_len head_dim -> bs seq_len (n_heads head_dim)")

        output = self.out(result)
        
        return output