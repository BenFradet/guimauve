import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))

    composed of
    - sequence encoder with positional information
    - normal embedding layer

    - integer encoding doesn't work since seq len is variable
    - [0, 1] encoding doesn't work since seq len is variable so intervals would be variable

    as a result, we need:
    - a bounded function that can generalize to seq of arbitrary len
    - a periodic function that can represent relative and absolute differences between positions
    consistently

    hence: sin or cos

    n=10k is large to increase wave length

    sin + cos because using cos at odd positions allows the model to learn relationships between
    relative positions
    i.e. for any offset k, PE(pos + k) is a function of PE(pos)
    [sin(pos + k)]   [cos(k)   sin(k)] [sin(pos)]
    [cos(pos + k)] = [-sin(k)  cos(k)] [cos(pos)]

    also, dot product of two positional encodings only depends on the distance k and it's symmetric

    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    - https://happystrongcoder.substack.com/p/transformer-with-code-part-i-positional
    """

    pos_encoding: torch.Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 512,
        seq_length: int = 100,
        n: int = 10000,
    ) -> None:
        """
        Args:
            num_embeddings: vocabulary size
            embedding_dim: dimension of the embedding vectors, must be even
            seq_length: maximum sequence length for positional encoding
            n: sinusoidal wavelength scaling
        """
        super(PositionalEmbedding, self).__init__()
        assert embedding_dim % 2 == 0, "Embedding dim needs to be even"
        self.token_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.seq_len = seq_length
        self.embedding_dim = embedding_dim
        self.embedding_dim_sqrt = math.sqrt(embedding_dim)
        self.register_buffer("pos_encoding", self._positional_encoding(n))

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normal embedding + positional encoding embedding

        Args:
            input: Tensor with shape [batch_size, seq_len]

        Returns:
            Tensor with shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = input.shape[1]
        embedded_tokens = self.token_embedding(input)
        scaled_tokens = embedded_tokens * self.embedding_dim_sqrt
        encoded_tokens = scaled_tokens + self.pos_encoding[None, :seq_len, :]
        return encoded_tokens

    def _positional_encoding(self, n: int = 10000) -> torch.Tensor:
        """
        Encodes position using sin/cos to represent order and absolute and relative distance
        information

        Args:
            n: constant for the sinusoidal functions to increase wavelength

        Returns:
            Tensor with shape [seq_len, embedding_dim]
        """
        half_embedding_dim = self.embedding_dim // 2
        pos = torch.reshape(
            torch.arange(start=0, end=self.seq_len, dtype=torch.float32), [-1, 1]
        )
        i = torch.reshape(
            torch.arange(start=0, end=half_embedding_dim, dtype=torch.float32), [1, -1]
        )
        denom = torch.pow(n, -i / half_embedding_dim)
        x = pos * denom
        sin = torch.sin(x).unsqueeze(-1)
        cos = torch.cos(x).unsqueeze(-1)
        encoding = torch.reshape(
            torch.concat([sin, cos], dim=-1), [self.seq_len, self.embedding_dim]
        )
        return encoding
