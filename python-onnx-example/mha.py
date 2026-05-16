import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    # c.f. https://happystrongcoder.substack.com/p/autoint-automatic-feature-interaction

    def __init__(
        self,
        embed_size,
        num_heads,
        key_dim=64,
        val_dim=None,
        dropout=0.0,
        use_bias=True,
        **kwargs,
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.key_output_dim = self.key_dim * num_heads
        self.val_output_dim = (
            self.val_dim * num_heads if val_dim else self.key_output_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

        self.weight_query = nn.Parameter(
            torch.empty((self.embed_size, self.key_output_dim))
        )
        self.weight_key = nn.Parameter(
            torch.empty((self.embed_size, self.key_output_dim))
        )
        self.weight_value = nn.Parameter(
            torch.empty((self.embed_size, self.val_output_dim))
        )
        self.weight_output = nn.Parameter(
            torch.empty((self.val_output_dim, self.embed_size))
        )

        if use_bias:
            self.bias_query = nn.Parameter(torch.empty(self.key_output_dim))
            self.bias_key = nn.Parameter(torch.empty(self.key_output_dim))
            self.bias_value = nn.Parameter(torch.empty(self.val_output_dim))
            self.bias_output = nn.Parameter(torch.empty(self.embed_size))
