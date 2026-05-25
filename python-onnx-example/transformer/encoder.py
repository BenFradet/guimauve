import torch
import torch.nn as nn

from transformer.feed_forward import FeedForward
from transformer.multi_head_attention import MultiHeadAttention
from transformer.residual_norm import ResidualNorm


class Encoder(nn.Module):
    """
    encoder is: mha -> residual norm -> ffn -> residual norm

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

        super(Encoder, self).__init__()
        assert model_dim % num_heads == 0, (
            "model_dim needs to be divisible by num_heads"
        )
        key_dim = model_dim // num_heads
        self.mha = MultiHeadAttention(
            embed_size=model_dim, num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.mha_residual_norm = ResidualNorm(dim=model_dim, dropout=dropout)
        self.ffn = FeedForward(model_dim=model_dim, feed_forward_dim=feed_forward_dim)
        self.ffn_residual_norm = ResidualNorm(dim=model_dim, dropout=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor with shape [batch_size, token_len, embedding_size]
        Returns:
            Tensor after mha, ffn and residual normalizations with shape
            [batch_size, token_len, embedding_size]
        """
        mha = self.mha(query=input, key=input, value=input)
        mha_residual_norm = self.mha_residual_norm(input, mha)
        ffn = self.ffn(mha_residual_norm)
        ffn_residual_norm = self.ffn_residual_norm(mha_residual_norm, ffn)
        return ffn_residual_norm
