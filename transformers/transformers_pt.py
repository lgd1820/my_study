from turtle import forward
import torch
import torch.nn as nn
from zmq import device

class Encoder(nn.Module):
    """
        Args:
            input_dim : 입력 차원
            hid_dim : 히든 차원
            n_layers : 레이어의 수
            n_heads : 헤드의 수
            pf_dim : Positionwise Feed Forwad 차원
            dropout : dropout
            device : 장치
            max_length : 최대 글자 수
    """
    def __init__(self, 
                input_dim,
                hid_dim,
                n_layers,
                n_heads,
                pf_dim,
                dropout,
                device,
                max_length = 100) -> None:
        super().__init__()

        # 학습 할 디바이스
        self.device = device
        
        # 토크나이즈 임베딩과 포지션 임베딩
        self.tokenize_embedding = nn.Embedding(input_dim, hid_dim)
        self.position_embedding = nn.Embedding(max_length, hid_dim)

        # n_layers 수 만큼 인코더 레이어 중첩
        self.layers = nn.ModuleList(
            [ EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    
    def forward(self, src, src_mask):
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1, 1, src_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch_size, src_len, hid_dim]
        src = self.dropout((self.tokenize_embedding(src) * self.scale) + self.position_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src

class EncoderLayer(nn.Module):
    """
        Args:
            hid_dim : 히든 차원
            n_heads : 헤드의 수
            pf_dim : Positionwise Feed Forwad 차원
            dropout : dropout
            device : 장치
    """
    def __init__(self,
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) -> None:
        super().__init__()

        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PostionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        # 셀프 어텐션
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, 잔차 연결, 정규화
        # src = [batch_size, src_len, hid_dim]
        src = self.self_attention_layer_norm(src + self.dropout(_src))

        # postionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, 잔차 연결, 정규화
        # src = [batch_size, src_len, hid_dim]
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src

class Decoder(nn.Module):
    def __init__(self,
                output_dim,
                hid_dim,
                n_layers,
                n_heads,
                pf_dim,
                dropout,
                device,
                max_length = 100) -> None:
        super().__init__()

        self.device = device
        self.tokenize_embedding = nn.Embedding(output_dim, hid_dim)
        self.position_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout((self.tokenize_embedding(trg) * self.scale) + self.position_embedding(pos))
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention
                
class DecoderLayer(nn.Module):
    """
        Args:
            hid_dim : 히든 차원
            n_heads : 헤드의 수
            pf_dim : Positionwise Feed Forwad 차원
            dropout : dropout
            device : 장치
    """
    def __init__(self,
                hid_dim,
                n_heads,
                pf_dim,
                dropout,
                device) -> None:
        super().__init__()
        
        self.self_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PostionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len, hid_dim]
        # enc_src = [batch_size, src_len, hid_dim]
        # trg_mask = [batch_size, 1, trg_len, trg_len]
        # src_mask = [batch_size, 1, 1, src_len] 

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, 잔차 연결, 정규화
        # src = [batch_size, trg_len, hid_dim]
        trg = self.self_attention_layer_norm(trg + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, 잔차 연결, 정규화
        # trg = [batch_size, trg_len, hid_dim]
        trg = self.encoder_attention_layer_norm(trg + self.dropout(_trg))

        # postionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, 잔차 연결, 정규화
        # trg = [batch_size, trg_len, hid_dim]
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class MultiHeadAttentionLayer(nn.Module):
    """
        다중 어텐션 행렬을 계산하기 위해 어텐션을 연결한 후 가중치 행렬을 곱해 행렬곱을 구한다.

        Args:
            hid_dim : 히든 차원
            n_heads : 헤드의 수
            dropout : dropout
            device : 장치
    """
    def __init__(self,
                hid_dim,
                n_heads,
                dropout,
                device) -> None:
        super().__init__()

        # 헤드의 수로 은닉 차원를 나눌 수 있는지
        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # 쿼리, 키, 밸류
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):
        """
            Args: 
                query : [bathc_size, query_len, hid_dim]
                key : [bathc_size, key_len, hid_dim]
                value : [bathc_size, value_len, hid_dim]
                mask : 디코더에서 사용되는 마스크드 멀티 헤드 어텐션을 위한 값
        """

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # reshape (batch_size, -1, n_heads, head_dim)
        # permute 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3
        # Q = [batch_size, n_heads, query_len, head_dim]
        # K = [batch_size, n_heads, key_len, head_dim]
        # V = [batch_size, n_heads, value_len, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # 쿼리 행렬과 키 행렬의 내적 연산을 수행 후 차원 수의 제곱근 값으로 나누기
        # 값을 나눔으로 인해서 안정적인 경사값을 얻을 수 있음
        # energy = [batch_size, n_heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 디코더의 마스크드 멀티 헤드 어텐션을 위한 마스크
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # 유사도 값을 소프트맥스 함수를 사용해 정규화
        # 이 행렬은 스코어 행렬
        # attention = [batch_size, n_heads, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)

        # 스코어 행렬과 밸류 행렬을 곱하여 어텐션 행렬을 구함
        # x = [batch_size, n_heads, query_len, head_dim]
        x = torch.matmul(self.dropout(attention), V)

        # permute 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3
        # x = [batch_size, query_len, n_heads, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch_size, query_len, hid_dim]
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention

class PostionwiseFeedforwardLayer(nn.Module):
    """
        Args:
            hid_dim : 히든 차원
            pf_dim : Positionwise Feed Forwad 차원
            dropout : dropout
    """
    def __init__(self,
                hid_dim,
                pf_dim,
                dropout) -> None:
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            피드포워드 네트워크는 2개의 dense layer와 ReLU 활성화 함수로 구성

            Args:
                x : 입력
            
            Returns:
                x : 2개의 dense layer와 ReLU 활성화 함수에 처리된 x 값
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self,
                encoder,
                decoder,
                src_pad_idx,
                trg_pad_idx,
                device) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch_size, trg_len]
        # trg_pad_mask = [batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
