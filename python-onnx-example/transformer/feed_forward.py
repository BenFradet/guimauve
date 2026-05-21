import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    sublayer: FFN(x) = max(0, xW1 + b1) W2 + b2 (relu)
    output: LayerNorm(x + Sublayer(x))
    """

    def __init__(
        self,
        feed_forward_dim: int = 2048,
        model_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super(FeedForward, self).__init__()
        self.feed_forward_dim = feed_forward_dim
        self.model_dim = model_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.model_dim)
        self.linear1 = nn.Linear(self.model_dim, self.feed_forward_dim)
        self.linear2 = nn.Linear(self.feed_forward_dim, self.model_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l1 = self.linear1(input).relu()
        l2 = self.linear2(l1)
        dropped_out = self.dropout(l2)
        # residual connection: result is added to the input
        summed = input + dropped_out
        normd = self.norm(summed)
        return normd
