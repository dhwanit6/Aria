"""
RMSNorm — Root Mean Square Layer Normalization.
Faster than LayerNorm (no mean computation, no bias).
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm as used in Llama, Gemma, and most modern LLMs."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in FP32 for numerical stability, cast back
        input_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(input_dtype) * self.weight
