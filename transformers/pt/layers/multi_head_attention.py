import torch
import torch.nn as nn

from scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads) -> None:
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):
        """
        헤드 수 만큼 텐서를 스플릿
            tensor: [batch_size, length, d_model]
            return: [batch_size, heads, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_heads
        tensor = tensor.view(batch_size, length, self.n_heads, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        스플릿된 텐서를 결합
            tensor: [batch_size, heads, length, d_tensor]
            return: [batch_size, length, d_model]
        """
        batch_size, heads, length, d_tensor = tensor.size()
        d_model = heads * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out
