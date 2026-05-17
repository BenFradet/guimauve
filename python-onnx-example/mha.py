import torch
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
        self.use_bias = bias

        self.query = nn.Linear(self.embed_size, self.key_output_dim, bias=self.use_bias)
        self.key = nn.Linear(self.embed_size, self.key_output_dim, bias=self.use_bias)
        self.value = nn.Linear(self.embed_size, self.val_output_dim, bias=self.use_bias)
        self.output = nn.Linear(self.val_output_dim, self.embed_size, bias=self.use_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor | None,
        key_mask: torch.Tensor | None,
        value_mask: torch.Tensor | None,
        use_causal_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return query, None

    def _compute_attention_mask(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        value_mask: torch.Tensor | None = None,
        use_causal_mask: bool = False,
    ) -> torch.Tensor | None:
        """
        Computes an attention mask, masks are combined using logical AND

        Args:
            query: Tensor of shape [B, T]
            value: Tensor of shape [B, S]
            query_mask: Tensor of shape [B, T]
            key_mask: Tensor of shape [B, S]
            value_mask: Tensor of shape [B, S]
            use_causal_mask: whether to use an additional causal mask with shape [1, T, S]
        Returns:
            Tensor with shape [B, T, S]
        """
        auto_mask: torch.Tensor | None = None
        if query_mask is not None:
            # B = batch size, T = max query length
            auto_mask = query_mask.unsqueeze(2)  # shape [B, T, 1]
        if value_mask is not None:
            # B = batch size, S = max value length
            mask = value_mask.unsqueeze(1)  # shape [B, 1, S]
            auto_mask = auto_mask & mask if auto_mask is not None else mask
        if key_mask is not None:
            # B = batch size, S = max key length = max value length
            mask = key_mask.unsqueeze(1)  # shape [B, 1, S]
            auto_mask = auto_mask & mask if auto_mask is not None else mask
        if use_causal_mask:
            # [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = auto_mask & mask if auto_mask is not None else mask
        return auto_mask

    def _compute_causal_mask(
        self,
        query: torch.Tensor,
        value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes a causal mask for masked self-attention layers

        Args:
            query: Tensor of shape [B, T]
            value: Tensor of shape [B, S], optional defaults to query
        Returns:
            a lower triangular Tensor with shape [1, T, S], e.g.
            ```
            [[True,  False],
              [True,  True]]
            ```
        """
        q_seq_len = query.shape[1]
        v_seq_len = value.shape[1] if value is not None else q_seq_len
        return torch.tril(torch.ones(1, q_seq_len, v_seq_len)).bool()
