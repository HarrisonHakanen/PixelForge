
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    DROPOUT,
    TEXT_EMBEDDING_DIM
)


class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        if D_MODEL % N_HEADS != 0:
            raise ValueError("D_MODEL deve ser divisível por N_HEADS.")

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


class CrossAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.head_dim = D_MODEL // N_HEADS

        self.query = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=False
        )

        self.key_value = nn.Linear(
            D_MODEL,
            D_MODEL * 2,
            bias=False
        )

        self.projection = nn.Linear(
            D_MODEL,
            D_MODEL
        )

        self.dropout = DROPOUT

    def forward(self, x, context, context_mask=None):

        batch_size, seq_length, _ = x.shape
        context_length = context.shape[1]

        q = self.query(x)

        kv = self.key_value(context)

        kv = kv.view(
            batch_size,
            context_length,
            2,
            N_HEADS,
            self.head_dim
        )

        q = q.view(
            batch_size,
            seq_length,
            N_HEADS,
            self.head_dim
        )

        q = q.transpose(1, 2)

        kv = kv.permute(2, 0, 3, 1, 4)

        k, v = kv.unbind(0)

        attention_mask = None

        if context_mask is not None:

            attention_mask = context_mask[:, None, None, :].bool()

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
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

        # Causal Self-Attention

        self.norm1 = nn.LayerNorm(D_MODEL)

        self.attention = CausalSelfAttention()

        # Cross-Attention

        self.norm2 = nn.LayerNorm(D_MODEL)

        self.cross_attention = CrossAttention()

        # Feed Forward

        self.norm3 = nn.LayerNorm(D_MODEL)

        self.feed_forward = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL * 4),
            nn.GELU(),
            nn.Linear(D_MODEL * 4, D_MODEL),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x, context, context_mask=None):

        # Consulta os tokens de pixels anteriores.

        x = x + self.attention(
            self.norm1(x)
        )

        # Consulta os embeddings textuais do CLIP.

        x = x + self.cross_attention(
            self.norm2(x),
            context,
            context_mask
        )

        # Processamento dos features.

        x = x + self.feed_forward(
            self.norm3(x)
        )

        return x


class PixelTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        # Pixel token embedding

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        # Positional embedding

        self.position_embedding = nn.Embedding(
            SEQUENCE_LENGTH,
            D_MODEL
        )

        # Projeta os embeddings do CLIP de 512 para 128.

        self.text_projection = nn.Linear(
            TEXT_EMBEDDING_DIM,
            D_MODEL
        )

        self.text_norm = nn.LayerNorm(D_MODEL)

        # Transformer blocks

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(N_LAYERS)
        ])

        self.norm = nn.LayerNorm(D_MODEL)

        # Output layer

        self.output = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False
        )

        # Weight tying

        self.output.weight = self.token_embedding.weight

    def forward(
        self,
        tokens,
        text_embeddings,
        text_attention_mask=None
    ):

        batch_size, seq_length = tokens.shape

        if seq_length > SEQUENCE_LENGTH:
            raise ValueError(
                "Sequência maior que o limite configurado."
            )

        if text_embeddings.shape[0] != batch_size:
            raise ValueError(
                "Batch de tokens e embeddings textuais incompatível."
            )

        if text_embeddings.shape[-1] != TEXT_EMBEDDING_DIM:
            raise ValueError(
                f"Esperado embedding textual com dimensão "
                f"{TEXT_EMBEDDING_DIM}, recebido "
                f"{text_embeddings.shape[-1]}."
            )

        if text_attention_mask is not None:
            if text_attention_mask.shape != text_embeddings.shape[:2]:
                raise ValueError(
                    "Dimensão da attention mask incompatível."
                )

        positions = torch.arange(
            seq_length,
            device=tokens.device
        ).unsqueeze(0)

        # Pixel embeddings

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )

        # Text embeddings

        context = self.text_projection(text_embeddings)

        context = self.text_norm(context)

        # Transformer

        for block in self.blocks:

            x = block(
                x,
                context,
                text_attention_mask
            )

        x = self.norm(x)

        return self.output(x)
