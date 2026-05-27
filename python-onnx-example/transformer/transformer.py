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

        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList(
            [
                Encoder(
                    model_dim=model_dim,
                    feed_forward_dim=feed_forward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    model_dim=model_dim,
                    feed_forward_dim=feed_forward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # converts decoder output into next token logits
        self.final_linear = nn.Linear(model_dim, target_vocab_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source_embedding = self.source_embedding(source)
        source_embedding = self.encoder_dropout(source_embedding)

        target_embedding = self.target_embedding(target)
        target_embedding = self.decoder_dropout(target_embedding)

        for encoder in self.encoders:
            source_embedding = encoder(source_embedding)

        for decoder in self.decoders:
            target_embedding = decoder(target_embedding, source_embedding)

        logits = self.final_linear(target_embedding)
        # softmax will be dealt with at inference or directly in the loss
        # probs = F.softmax(logits, dim=-1)

        return logits
