import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    # c.f. https://happystrongcoder.substack.com/p/autoint-automatic-feature-interaction

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        key_dim: int = 64,
        val_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.key_output_dim = self.key_dim * num_heads
        self.val_output_dim = (
            self.val_dim * num_heads if self.val_dim else self.key_output_dim
        )
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(self.embed_size, self.key_output_dim, bias=bias)
        self.key = nn.Linear(self.embed_size, self.key_output_dim, bias=bias)
        self.value = nn.Linear(self.embed_size, self.val_output_dim, bias=bias)
        self.output = nn.Linear(self.val_output_dim, self.embed_size, bias=bias)
