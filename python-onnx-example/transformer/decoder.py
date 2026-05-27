import torch
import torch.nn as nn

from transformer.feed_forward import FeedForward
from transformer.multi_head_attention import MultiHeadAttention
from transformer.residual_norm import ResidualNorm


class Decoder(nn.Module):
    """
    decoder: masked mha -> residual norm -> cross mha -> residual norm -> ffn -> residual norm

    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    - https://happystrongcoder.substack.com/p/transformer-with-code-part-ii-encoder
    """

    def __init__(
        self,
        model_dim: int = 512,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            model_dim: dimension of the model (input/output), must be divisible by num_heads
            feed_forward_dim: inner dimension of the feed-forward sublayer
            num_heads: number of parallel attention heads
            dropout: dropout probability applied after attention and feed-forward sublayers
        """

        super(Decoder, self).__init__()
        assert model_dim % num_heads == 0, (
            "model_dim needs to be divisible by num_heads"
        )
        key_dim = model_dim // num_heads

        self.masked_mha = MultiHeadAttention(
            embed_size=model_dim, num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.masked_mha_residual_norm = ResidualNorm(dim=model_dim, dropout=dropout)

        self.cross_mha = MultiHeadAttention(
            embed_size=model_dim, num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.cross_mha_residual_norm = ResidualNorm(dim=model_dim, dropout=dropout)

        self.ffn = FeedForward(model_dim=model_dim, feed_forward_dim=feed_forward_dim)
        self.ffn_residual_norm = ResidualNorm(dim=model_dim, dropout=dropout)

    def forward(
        self, input: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input: Tensor with shape [batch_size, target_len, embedding_size]
            encoder_output: Tensor with shape [batch_size, source_len, embedding_size], used for key
            and value of cross mha
        Returns:
            Tensor after masked mha, cross mha and ffn with shape
            [batch_size, token_len, embedding_size]
        """
        masked_mha = self.masked_mha(
            query=input,
            key=input,
            value=input,
            use_causal_mask=True,
        )
        masked_mha_residual_norm = self.masked_mha_residual_norm(
            sublayer_input=input,
            sublayer_output=masked_mha,
        )

        cross_mha = self.cross_mha(
            query=masked_mha_residual_norm,
            key=encoder_output,
            value=encoder_output,
        )
        cross_mha_residual_norm = self.cross_mha_residual_norm(
            sublayer_input=masked_mha_residual_norm,
            sublayer_output=cross_mha,
        )

        ffn = self.ffn(cross_mha_residual_norm)
        ffn_residual_norm = self.ffn_residual_norm(
            sublayer_input=cross_mha_residual_norm,
            sublayer_output=ffn,
        )

        return ffn_residual_norm
