"""
Shared Sliding-Window GQA Attention with RoPE.

Key design decisions:
- Grouped-Query Attention (GQA): fewer KV heads than Q heads → smaller KV cache
- Rotary Position Embeddings (RoPE): relative position encoding, proven at scale
- Sliding window: only attend to last W tokens → O(W) memory, not O(N)
- Shared weights: all attention checkpoints share ONE weight set (Zamba trick)
  → 6 attention layers cost ≈ 1 layer of parameters

Reference: GQA (Ainslie et al., 2023), RoPE (Su et al., 2021), Zamba (Glorioso et al., 2024)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import LinearFlex
from .rmsnorm import RMSNorm


def precompute_rope_frequencies(
    d_head: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for RoPE.
    
    RoPE applies a rotation to pairs of dimensions in Q and K:
        q'_{2i}   = q_{2i} cos(mθ_i) - q_{2i+1} sin(mθ_i)
        q'_{2i+1} = q_{2i} sin(mθ_i) + q_{2i+1} cos(mθ_i)
    
    where θ_i = 1 / 10000^(2i/d) and m is the position index.
    
    Returns:
        cos_cached: [max_seq_len, d_head//2]
        sin_cached: [max_seq_len, d_head//2]
    """
    # Frequency bands: θ_i = 1 / 10000^(2i/d)
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    # Position indices
    positions = torch.arange(max_seq_len, device=device).float()
    # Outer product: [seq_len, d_head//2]
    angles = torch.outer(positions, freqs)
    
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    Apply RoPE to a tensor x of shape [batch, n_heads, seq_len, d_head].
    
    Uses the interleaved approach: pairs (x[..., 0], x[..., 1]),
    (x[..., 2], x[..., 3]), etc.
    """
    T = x.shape[2]
    d_half = x.shape[-1] // 2
    
    # Select the right position indices
    cos_t = cos[position_offset : position_offset + T, :d_half]  # [T, d_half]
    sin_t = sin[position_offset : position_offset + T, :d_half]
    
    # Reshape for broadcasting: [1, 1, T, d_half]
    cos_t = cos_t.unsqueeze(0).unsqueeze(0)
    sin_t = sin_t.unsqueeze(0).unsqueeze(0)
    
    # Split into pairs
    x_even = x[..., 0::2]  # [B, H, T, d_half]
    x_odd = x[..., 1::2]
    
    # Apply rotation
    x_even_rot = x_even * cos_t - x_odd * sin_t
    x_odd_rot = x_even * sin_t + x_odd * cos_t
    
    # Interleave back
    out = torch.stack([x_even_rot, x_odd_rot], dim=-1)
    return out.flatten(-2)  # [B, H, T, d_head]


class SlidingWindowGQA(nn.Module):
    """
    Grouped-Query Attention with sliding window and RoPE.
    
    GQA: n_kv_heads < n_heads. Each KV head is shared across
    (n_heads // n_kv_heads) query heads. Reduces KV cache size.
    
    Sliding window: each query only attends to the last `window_size`
    keys/values. This bounds memory and forces the model to use
    the RWKV state for long-range dependencies (complementary design).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        window_size: int = 512,
        max_seq_len: int = 2048,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.window_size = window_size
        self.n_groups = n_heads // n_kv_heads  # Q heads per KV head

        self.scale = 1.0 / math.sqrt(d_head)

        # Projections
        self.W_q = LinearFlex(d_model, n_heads * d_head, use_ternary=use_ternary)
        self.W_k = LinearFlex(d_model, n_kv_heads * d_head, use_ternary=use_ternary)
        self.W_v = LinearFlex(d_model, n_kv_heads * d_head, use_ternary=use_ternary)
        self.W_o = LinearFlex(d_model, d_model, use_ternary=use_ternary)

        # Precompute RoPE tables (registered as buffers → saved with model, moved with .to())
        cos, sin = precompute_rope_frequencies(d_head, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]
            kv_cache: (k_cache, v_cache) each [batch, n_kv_heads, cache_len, d_head]
            position_offset: absolute position of x[0] (for RoPE in generation mode)
            is_causal: whether to apply causal mask

        Returns:
            output: [batch, seq_len, d_model]
            new_kv_cache: updated (k_cache, v_cache)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # q: [B, n_heads, T, d_head]
        # k, v: [B, n_kv_heads, T, d_head]

        # Apply RoPE to Q and K
        q = apply_rope(q, self.rope_cos, self.rope_sin, position_offset)
        k = apply_rope(k, self.rope_cos, self.rope_sin, position_offset)

        # Update KV cache (inference only: append to existing cache)
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        S = k.shape[2]  # total key sequence length

        # Sliding window: ONLY trim during inference (T=1).
        # During training (T>1), we keep all KV and use a banded causal mask.
        # Trimming during training causes NaN: early queries would have
        # zero valid keys after causal masking of the trimmed-to-end KV.
        if T == 1 and S > self.window_size:
            k = k[:, :, -self.window_size:, :]
            v = v[:, :, -self.window_size:, :]
            S = self.window_size

        # Cache the current KV (trimmed to window for memory)
        cache_k = k[:, :, -self.window_size:, :].detach() if S > self.window_size else k.detach()
        cache_v = v[:, :, -self.window_size:, :].detach() if S > self.window_size else v.detach()
        new_kv_cache = (cache_k, cache_v)

        # Expand KV heads for GQA: repeat each KV head for its group of Q heads
        # k: [B, n_kv_heads, S, d_head] → [B, n_heads, S, d_head]
        k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        k = k.reshape(B, self.n_heads, -1, self.d_head)
        v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
        v = v.reshape(B, self.n_heads, -1, self.d_head)

        S = k.shape[2]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, S]

        # Sliding window causal mask
        if is_causal and T > 1:
            idx_q = torch.arange(T, device=x.device) + (S - T)
            idx_k = torch.arange(S, device=x.device)
            # Causal: j <= i AND Sliding window: j >= i - window + 1
            mask = (idx_k.unsqueeze(0) <= idx_q.unsqueeze(1)) & \
                   (idx_k.unsqueeze(0) >= idx_q.unsqueeze(1) - self.window_size + 1)
            attn = attn.masked_fill(~mask.view(1, 1, T, S), float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # [B, H, T, d_head]
        out = out.transpose(1, 2).reshape(B, T, self.d_model)

        # Output projection
        out = self.W_o(out)

        return out, new_kv_cache


class SharedAttentionBlock(nn.Module):
    """
    Full attention block: RMSNorm → SlidingWindowGQA → Residual.
    
    The FFN is handled externally (same MoE as RWKV layers).
    
    'Shared' means multiple instances of this block reference the SAME
    underlying SlidingWindowGQA weights. This is handled at the model
    assembly level (aria.py), not here.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        window_size: int = 512,
        max_seq_len: int = 2048,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attention = SlidingWindowGQA(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            window_size=window_size,
            max_seq_len=max_seq_len,
            use_ternary=use_ternary,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = x
        x_norm = self.norm(x)
        attn_out, new_kv_cache = self.attention(
            x_norm, kv_cache, position_offset, is_causal=True
        )
        return residual + attn_out, new_kv_cache
