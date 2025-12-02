# src/transformer.py
import math, torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq, d_model]
        return x + self.pe[:, : x.size(1)]

class TransformerENFR(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 d_model=256, nhead=4, layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.src_tok = nn.Embedding(len(src_vocab), d_model, padding_idx=src_vocab["<pad>"])
        self.tgt_tok = nn.Embedding(len(tgt_vocab), d_model, padding_idx=tgt_vocab["<pad>"])
        self.pos = PositionalEncoding(d_model)
        self.tr = nn.Transformer(d_model, nhead,
                                 num_encoder_layers=layers,
                                 num_decoder_layers=layers,
                                 dim_feedforward=d_ff,
                                 dropout=dropout,
                                 batch_first=True)
        self.out = nn.Linear(d_model, len(tgt_vocab))
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

    def encode(self, src, src_mask):
        return self.tr.encoder(self.pos(self.src_tok(src)),
                               src_key_padding_mask=src_mask)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
    
        # 1) Encode
        memory = self.encode(src, src_pad_mask)

        # 2) Build a causal (look-ahead) mask for decoder self-attention
        seq_len = tgt.size(1)
        causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device), 1)

        # 3) Decode
        out = self.tr.decoder(
            self.pos(self.tgt_tok(tgt)),        # embedded + positional
            memory,
            tgt_mask=causal,                    # prevents peeking ahead
            tgt_key_padding_mask=tgt_pad_mask,  # ignores PADs in tgt
            memory_key_padding_mask=src_pad_mask)

        return self.out(out)                    # final linear layer

