"""
BitLinear — Ternary weight linear layer (BitNet b1.58).
During training: weights are FP32, quantized via STE (Straight-Through Estimator).
During inference: weights are pre-quantized to {-1, 0, +1}, no multiplications.

Reference: "The Era of 1-bit LLMs" (Microsoft, 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def ternary_quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to {-1, 0, +1} using absmean scaling.
    Returns (quantized_weights, scale_factor).
    """
    # Scale factor: mean of absolute values
    gamma = w.abs().mean().clamp(min=1e-8)
    # Scale to roughly unit range and round to {-1, 0, 1}
    w_scaled = w / gamma
    w_ternary = w_scaled.clamp(-1, 1).round()
    return w_ternary, gamma


def activation_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to INT8 range for integer-only accumulation.
    Per-token quantization (each token gets its own scale).
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_quant = (x * scale).round().clamp(-128, 127)
    x_dequant = x_quant / scale
    return x_dequant, scale


class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with ternary weights.
    
    Training: FP32 weights, quantized on-the-fly via STE.
    Inference: Pre-quantized {-1, 0, 1} weights, no multiplications.
    
    The STE trick: round() in forward pass (non-differentiable),
    but gradient flows through as if round() were identity.
    This is achieved via: w + (w_quant - w).detach()
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Latent FP32 weights (learnable, what the optimizer updates)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        # Initialize with scaled normal (adapted for ternary convergence)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with STE for ternary weights.
        
        Math:
            W_ternary = round(clamp(W / γ_w, -1, 1))     ∈ {-1, 0, 1}
            Y = X @ (W_ternary * γ_w)^T
            
        STE: gradients w.r.t. W flow through round() as identity.
        
        Note: Activation quantization (INT8) is ONLY used during inference
        export, not during training. During training we do FP32 matmul
        anyway (the STE makes it equivalent), and the x.abs().max()
        operation in activation_quant creates autograd instabilities.
        """
        # Weight quantization with STE
        w_ternary, w_scale = ternary_quantize(self.weight)
        # STE: forward uses quantized, backward uses latent
        w_ste = self.weight + (w_ternary * w_scale - self.weight).detach()
        
        # Standard matmul with STE weights (activations stay FP32)
        y = F.linear(x, w_ste, self.bias)
        
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, ternary=True")


class LinearFlex(nn.Module):
    """
    Wrapper that switches between BitLinear (ternary) and standard nn.Linear.
    Use config.use_ternary to control globally.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, use_ternary: bool = True):
        super().__init__()
        if use_ternary:
            self.linear = BitLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
