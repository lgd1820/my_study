import torch
import torch.nn as nn

from module.bert import BERT

class BERTPretrain(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout, segment_type) -> None:
        super().__init__()
        self.bert = BERT(vocab_size, num_layers, d_model, n_heads, hidden, max_len, dropout, segment_type)

        self.projection_cls = nn.Linear(d_model, 2, bias=False)
        self.projection_lm = nn.Linear(d_model, vocab_size, bias=False)
        self.projection_lm.weight = self.bert.encoder.embedding.weight

    def forward(self, inputs, segments):
        outputs, outputs_cls, attn_probs = self.bert(inputs, segments)
        logits_cls = self.projection_cls(outputs_cls)
        logits_lm = self.projection_lm(outputs)

        return logits_cls, logits_lm, attn_probs