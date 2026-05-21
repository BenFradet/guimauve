import torch.nn as nn


class FeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1) W2 + b2 (relu)
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
