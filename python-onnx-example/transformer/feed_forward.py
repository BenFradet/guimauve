import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Fully connected feed forward network
    FFN(x) = max(0, xW1 + b1) W2 + b2 (relu)

    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    - https://happystrongcoder.substack.com/p/transformer-with-code-part-ii-encoder
    """

    def __init__(
        self,
        model_dim: int = 512,
        feed_forward_dim: int = 2048,
    ) -> None:
        """
        Args:
            model_dim: input and output dimension, hyper parameter 
            feed_forward_dim: inner layer dimension, expansion factor, hyper parameter
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(model_dim, feed_forward_dim)
        self.linear2 = nn.Linear(feed_forward_dim, model_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor fed through the network with shape [batch_size, seq_len, model_dim]

        Returns:
            Tensor with shape [batch_size, seq_len, model_dim]
        """
        l1 = self.linear1(input).relu()
        return self.linear2(l1)
