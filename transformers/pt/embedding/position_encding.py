import torch
import torch.nn as nn

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