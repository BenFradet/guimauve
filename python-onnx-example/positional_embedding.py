import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    - https://happystrongcoder.substack.com/p/transformer-with-code-part-i-positional
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int = 512,
        length: int = 100,
        n: int = 10000,
    ) -> None:
        super(PositionalEmbedding, self).__init__()
        assert embedding_dim % 2 == 0, "Embedding dim needs to be even"
        self.token_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.len = length
        self.embedding_dim = embedding_dim
        self.register_buffer('pos_encoding', self._positional_encoding(n))

    def _positional_encoding(self, n: int = 10000) -> torch.Tensor:
        half_embedding_dim = self.embedding_dim // 2
        pos = torch.reshape(
            torch.arange(start=0, end=self.len, dtype=torch.float32), [-1, 1]
        )
        i = torch.reshape(
            torch.arange(start=0, end=half_embedding_dim, dtype=torch.float32), [1, -1]
        )
        denom = torch.pow(n, -i / half_embedding_dim)
        x = pos * denom
        sin = torch.sin(x).unsqueeze(-1)
        cos = torch.cos(x).unsqueeze(-1)
        encoding = torch.reshape(
            torch.concat([sin, cos], dim=-1), [self.len, self.embedding_dim]
        )
        return encoding
