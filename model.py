import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DROPOUT
)


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.head_dim = D_MODEL // N_HEADS

        self.qkv = nn.Linear(
            D_MODEL,
            D_MODEL * 3
        )

        self.projection = nn.Linear(
            D_MODEL,
            D_MODEL
        )

        self.dropout = DROPOUT

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            seq_length,
            3,
            N_HEADS,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            seq_length,
            D_MODEL
        )

        return self.projection(output)


class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(D_MODEL)
        self.attention = CausalSelfAttention()

        self.norm2 = nn.LayerNorm(D_MODEL)

        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 4),
            nn.GELU(),
            nn.Linear(D_MODEL * 4, D_MODEL),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class PixelTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        self.position_embedding = nn.Embedding(
            SEQUENCE_LENGTH,
            D_MODEL
        )

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(N_LAYERS)
        ])

        self.norm = nn.LayerNorm(D_MODEL)

        self.output = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False
        )

        self.output.weight = self.token_embedding.weight

    def forward(self, tokens):
        batch_size, seq_length = tokens.shape

        if seq_length > SEQUENCE_LENGTH:
            raise ValueError("Sequência maior que o limite configurado.")

        positions = torch.arange(
            seq_length,
            device=tokens.device
        ).unsqueeze(0)

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.output(x)