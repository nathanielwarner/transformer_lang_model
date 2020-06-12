import math
import torch
import torch.nn as nn
import json
from keras_preprocessing.sequence import pad_sequences
import numpy as np


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
        self.input_length = input_length

    @classmethod
    def from_description(cls, path):
        with open(path) as json_file:
            model_description = json.load(json_file)
        vocab_size = model_description["vocab_size"]
        input_length = model_description["input_length"]
        d_model = model_description["d_model"]
        n_heads = model_description["n_heads"]
        d_ff = model_description["d_ff"]
        n_layers = model_description["n_layers"]
        dropout = model_description["dropout"]
        return cls(vocab_size, input_length, d_model, n_heads, d_ff, n_layers, dropout=dropout)

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


def predict(prompt, transformer, device, preprocess, tokenizer, max_out_len):
    prompt = preprocess(prompt)
    prompt_tok = tokenizer.EncodeAsIds(prompt)
    len_prompt = len(prompt_tok)
    prompt_padded = pad_sequences([prompt_tok], maxlen=transformer.input_length, padding='post', truncating='post',
                                  value=0, dtype='int64')
    out_toks = []
    for j in range(max_out_len):
        out = transformer(torch.tensor(prompt_padded).to(device))[0]
        working_idx = min(len_prompt - 1 + j, transformer.input_length - 1)
        new_tok = torch.argmax(out[working_idx]).item()
        out_toks.append(new_tok)
        if new_tok == tokenizer.eos_id():
            break
        if len_prompt + j < transformer.input_length:
            prompt_padded[0][len_prompt + j] = new_tok
        else:
            prompt_padded = np.roll(prompt_padded, -1, axis=1)
            prompt_padded[0][-1] = new_tok
    out_detok = tokenizer.DecodeIds(out_toks)
    return out_detok
