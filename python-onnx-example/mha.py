import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    attention: a(Q, {K: V}) = σ(Q @ K^T / sqrt(dk)) @ V
    multi-head: mha(Q, {K; V}) = concat(h_i)W^o, h_i = a(Q @ W^Q_i, {K @ W^K_i: V @ W^V_i})

    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    - https://happystrongcoder.substack.com/p/autoint-automatic-feature-interaction
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        key_dim: int = 64,
        val_dim: int | None = None,
        dropout: float = 0.0,
        use_bias: bool = True,
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

        self.query = nn.Linear(self.embed_size, self.key_output_dim, bias=use_bias)
        self.key = nn.Linear(self.embed_size, self.key_output_dim, bias=use_bias)
        self.value = nn.Linear(self.embed_size, self.val_output_dim, bias=use_bias)
        self.output = nn.Linear(self.val_output_dim, self.embed_size, bias=use_bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_mask: torch.Tensor | None,
        key_mask: torch.Tensor | None,
        value_mask: torch.Tensor | None,
        use_causal_mask: bool = False,
    ) -> torch.Tensor:
        """
        B: batch size
        S: key len, source seq len
        T: query len, target seq len

        Args:
            query: Tensor with shape [B, T, embed_size]
            key: Tensor with shape [B, S, embed_size]
            value: Tensor with shape [B, S, embed_size]
            query_mask: Tensor of shape [B, T]
            key_mask: Tensor of shape [B, S]
            value_mask: Tensor of shape [B, S]
            use_causal_mask: whether to use an additional causal mask with shape [1, T, S]

        Returns:
            Tensor with shape [B, T, embed_size]
        """
        # [B, T, S]
        mask = self._compute_attention_mask(
            query, value, query_mask, key_mask, value_mask, use_causal_mask
        )
        # [B, T, key_dim * num_heads]
        queries = self.query(query)
        # [B, S, key_dim * num_heads]
        keys = self.key(key)
        # [B, S, val_dim * num_heads]
        values = self.value(value)

        # [num_heads, B, T, key_dim]
        queries = torch.stack(queries.chunk(self.num_heads, dim=2), dim=0)
        # [num_heads, B, S, key_dim]
        keys = torch.stack(keys.chunk(self.num_heads, dim=2), dim=0)
        # [num_heads, B, S, val_dim]
        values = torch.stack(values.chunk(self.num_heads, dim=2), dim=0)

        # [num_heads, B, T, S]
        # attention scaling: sqrt(key_dim)
        weights = torch.matmul(queries, keys.transpose(-2, -1)) / (self.key_dim ** 0.5)
        if mask is not None:
            weights += (1.0 - mask.type(weights.dtype)) * -1e9

        scores = F.softmax(weights, dim=-1)
        scores = self.dropout(scores)

        # [num_heads, B, T, val_dim]
        outputs = torch.matmul(scores, values)
        # list of num_heads [1, B, T, val_dim] tensors
        outputs = torch.split(outputs, 1, dim=0)
        # [1, B, T, val_dim * num_heads]
        outputs = torch.concat(outputs, dim=-1)
        # [B, T, val_dim * num_heads]
        outputs = torch.squeeze(outputs, dim=0)

        outputs = self.output(outputs)

        return outputs

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
