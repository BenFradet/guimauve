import torch
import torch.nn as nn

from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.positional_embedding import PositionalEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        seq_length: int,
        num_layers: int = 6,
        model_dim: int = 512,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super(Transformer, self).__init__()
        assert model_dim % num_heads == 0, (
            "model_dim needs to be divisible by num_heads"
        )

        self.source_embedding = PositionalEmbedding(
            num_embeddings=source_vocab_size,
            embedding_dim=model_dim,
            seq_length=seq_length,
        )
        self.target_embedding = PositionalEmbedding(
            num_embeddings=target_vocab_size,
            embedding_dim=model_dim,
            seq_length=seq_length,
        )

        self.encoders = nn.ModuleList([
            Encoder(
                model_dim=model_dim,
                feed_forward_dim=feed_forward_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.encoder_dropout = nn.Dropout(dropout)

        self.decoders = nn.ModuleList([
            Decoder(
                model_dim=model_dim,
                feed_forward_dim=feed_forward_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.decoder_dropout = nn.Dropout(dropout)

        # converted decoder output into next token probabilities (to be used with softmax)
        self.final_linear = nn.Linear(model_dim, target_vocab_size)
