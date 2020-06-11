import math
import torch
import torch.nn as nn


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, input_length, d_model, n_heads, d_ff, n_layers, dropout=0.5):
        super(TransformerLM, self).__init__()
        self.model_type = 'Transformer_LM'
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.coord_emb = CoordinateEmbedding(d_model, max_len=max(input_length, n_layers))
        self.encoder = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff, dropout=dropout)
        self.n_core_iter = n_layers
        self._init_weights()
        self.src_mask = None
        self.scale_factor = math.sqrt(d_model)
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):

        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask

        src = self.embedding(src).transpose(0, 1) * self.scale_factor

        x = src
        for time_step in range(self.n_core_iter):
            x = self.coord_emb(x, time_step)
            x = self.encoder(x, src_mask=self.src_mask)

        out = x.transpose(0, 1)
        final = self.final_layer(out)
        return final


class CoordinateEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(CoordinateEmbedding, self).__init__()

        ce = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        ce[:, 0::2] = torch.sin(position * div_term)
        ce[:, 1::2] = torch.cos(position * div_term)
        ce = ce.unsqueeze(0).transpose(0, 1)
        self.register_buffer('ce', ce)

    def forward(self, x, time_step):
        x = x + self.ce[:x.size(0), :]
        x = x + self.ce[0, 0, time_step]
        return x



