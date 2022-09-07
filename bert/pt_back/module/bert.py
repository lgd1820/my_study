import torch
import torch.nn as nn

from module.encoder import Encoder
from module.utils import make_mask

class BERT(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout, segment_type) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout, segment_type)

        self.fc = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    
    def forword(self, inputs, segments):
        enc_padding_mask = make_mask(inputs, "padding")

        outputs, attn_probs = self.encoder(inputs, segments, enc_padding_mask)

        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.fc(outputs_cls)
        outputs_cls = self.tanh(outputs_cls)

        return outputs, outputs_cls, attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dic": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dic"])

        return save["epoch"], save["loss"]