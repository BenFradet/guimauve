import torch
import torch.nn as nn


class ResidualNorm(nn.Module):
    """
    Handles drop out, residual connection and normalization after each sublayer
    LayerNorm(x + Sublayer(x))

    c.f.
    - attention is all you need https://arxiv.org/pdf/1706.03762
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            dim: input / output dimension
            dropout: dropout probability applied to the output
        """
        super(ResidualNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, sublayer_input: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sublayer_input: original input before the sublayer, aka residual with shape [*, dim]
            sublayer_output: output of the sublayer with shape [*, dim]

        Returns:
            Tensor with shape [*, dim]
        """
        dropped_out = self.dropout(sublayer_output)
        # residual connection: input is added to the sublayer output
        summed = sublayer_input + dropped_out
        normd = self.norm(summed)
        return normd
