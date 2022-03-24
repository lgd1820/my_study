import torch.nn as nn

from module.encoder import Encoder
from module.decoder import Decoder
from module.utils import make_mask

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout) -> None:
        super().__init__()
        self.enc_outputs = Encoder(vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout)
        self.dec_outputs = Decoder(vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout)
        self.output = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, dec_input):
        enc_input = input
        dec_input = dec_input
        enc_padding_mask = make_mask(enc_input, "padding")
        dec_padding_mask = make_mask(enc_input, "padding")
        look_ahead_mask = make_mask(dec_input, "lookahead")

        enc_output = self.enc_outputs(enc_input, enc_padding_mask)
        dec_output = self.dec_outputs(enc_output, dec_input, dec_padding_mask, look_ahead_mask)

        outputs = self.output(dec_output)

        return outputs