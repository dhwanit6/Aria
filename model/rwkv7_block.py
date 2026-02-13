"""
RWKV-7 Time Mixing Block with Generalized Delta Rule.

The core recurrence:
    S_t = decay * S_{t-1} - η * (S_{t-1} @ k_t - v_t) ⊗ k_t^T
    wkv_t = S_t @ r_t
    out_t = gate * O(wkv_t)

Where S is the state matrix, r/k/v are receptance/key/value,
and the delta rule provides in-context learning via state adaptation.

Reference: RWKV-7 "Goose" (Peng et al., 2024)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import LinearFlex
from .rmsnorm import RMSNorm


class RWKV7TimeMixing(nn.Module):
    """RWKV-7 time mixing with delta rule state update."""

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 use_ternary: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        # Linear projections (ternary via BitLinear)
        self.W_r = LinearFlex(d_model, d_model, use_ternary=use_ternary)  # Receptance
        self.W_k = LinearFlex(d_model, d_model, use_ternary=use_ternary)  # Key
        self.W_v = LinearFlex(d_model, d_model, use_ternary=use_ternary)  # Value
        self.W_g = LinearFlex(d_model, d_model, use_ternary=use_ternary)  # Gate
        self.W_o = LinearFlex(d_model, d_model, use_ternary=use_ternary)  # Output

        # Delta rule learnable parameters (per-head)
        # eta: learning rate for state correction
        self.eta = nn.Parameter(torch.zeros(n_heads, 1, 1))
        # decay: exponential decay of state (initialized near 1.0 for long memory)
        self.decay = nn.Parameter(torch.zeros(n_heads, 1, 1))

        # Token-shift mixing (temporal interpolation between current and previous token)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_g = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        # Group norm on wkv output (stabilizes ternary training)
        self.group_norm = nn.GroupNorm(
            num_groups=n_heads, num_channels=d_model, eps=1e-5, affine=True
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize decay close to 1 (log scale, will be sigmoid'd)
        # Higher decay = longer memory, more stable
        nn.init.uniform_(self.decay, 2.0, 4.0)  # sigmoid → ~0.88 to 0.98
        # Initialize eta VERY small (conservative state updates)
        # Too-large eta causes state explosion via accumulating outer products
        nn.init.uniform_(self.eta, -3.0, -1.0)  # sigmoid → ~0.05 to 0.27

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        prev_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both training (sequential over time) and inference.

        Args:
            x: [batch, seq_len, d_model] for training, [batch, 1, d_model] for inference
            state: [batch, n_heads, d_head, d_head] — the S matrix (None → init zeros)
            prev_x: [batch, 1, d_model] — previous timestep's x for token shift

        Returns:
            output: [batch, seq_len, d_model]
            new_state: [batch, n_heads, d_head, d_head]
            last_x: [batch, 1, d_model] — last x for next call's token shift
        """
        B, T, C = x.shape

        # Initialize state if needed
        if state is None:
            state = torch.zeros(
                B, self.n_heads, self.d_head, self.d_head,
                device=x.device, dtype=x.dtype
            )

        # Token shift: interpolate between current and previous token
        if prev_x is None:
            prev_x = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)

        # Shift: prepend prev_x, drop last
        x_shifted = torch.cat([prev_x, x[:, :-1, :]], dim=1)

        # Mix current and shifted
        x_r = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)
        x_k = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        x_v = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        x_g = x * self.time_mix_g + x_shifted * (1 - self.time_mix_g)

        # Projections
        r = torch.sigmoid(self.W_r(x_r))  # [B, T, C] — receptance (gate)
        k = self.W_k(x_k)                 # [B, T, C] — key
        v = self.W_v(x_v)                 # [B, T, C] — value
        g = F.silu(self.W_g(x_g))         # [B, T, C] — output gate

        # Reshape for multi-head: [B, T, n_heads, d_head]
        r = r.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # L2-normalize keys AND values per-head — CRITICAL for delta rule stability.
        # Without this, ||S @ k|| grows unbounded as state accumulates
        # outer products, causing NaN within ~30 tokens.
        k = F.normalize(k, p=2, dim=-1)  # unit norm per head
        v = F.normalize(v, p=2, dim=-1)  # bound correction magnitude

        # Delta rule parameters
        decay = torch.sigmoid(self.decay)  # [n_heads, 1, 1] ∈ (0, 1)
        eta = torch.sigmoid(self.eta)      # [n_heads, 1, 1] ∈ (0, 1)

        # State norm bound: sqrt(d_head) is the natural scale for a D×D matrix
        max_state_norm = math.sqrt(self.d_head) * 2.0

        # Process tokens sequentially (the recurrent scan)
        outputs = []
        for t in range(T):
            # RWKV-7 recurrence step
            r_t = r[:, t, :, :]  # [B, H, D]
            k_t = k[:, t, :, :]  # [B, H, D]
            v_t = v[:, t, :, :]  # [B, H, D]

            # Delta rule state update (computed in float32 for precision)
            k_col = k_t.unsqueeze(-1).float()  # [B, H, D, 1]
            v_t_f = v_t.float()
            state = state.float()

            # prediction: S @ k → what the state thinks v should be
            pred = torch.matmul(state, k_col).squeeze(-1)  # [B, H, D]

            # error: prediction - actual (FP32 to avoid catastrophic cancellation)
            error = pred - v_t_f  # [B, H, D]

            # correction: outer product of error and k
            correction = eta * error.unsqueeze(-1) * k_col.transpose(-2, -1)  # [B, H, D, D]

            # state update: decay old state, apply correction
            state = decay * state - correction

            # Frobenius norm bounding (differentiable, unlike clamp)
            state_norm = state.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            scale_factor = torch.where(
                state_norm > max_state_norm,
                max_state_norm / state_norm,
                torch.ones_like(state_norm),
            )
            state = state * scale_factor
            state = state.to(x.dtype)

            # output: S @ r (what the state retrieves for receptance)
            r_col = r_t.unsqueeze(-1)      # [B, H, D, 1]
            wkv = torch.matmul(state, r_col).squeeze(-1)  # [B, H, D]

            outputs.append(wkv)

        # Stack outputs: [B, T, H, D]
        wkv_out = torch.stack(outputs, dim=1)
        # Reshape to [B, T, C] and apply group norm
        wkv_out = wkv_out.reshape(B, T, self.d_model)
        wkv_out = self.group_norm(wkv_out.transpose(1, 2)).transpose(1, 2)

        # Output projection with gate
        out = self.W_o(wkv_out) * g

        # Save last x for next call's token shift
        last_x = x[:, -1:, :]

        return out, state, last_x


class RWKV7Block(nn.Module):
    """
    Full RWKV-7 block: RMSNorm → TimeMixing → Residual → RMSNorm → FFN → Residual

    Note: The FFN is handled externally (MoE or dense, chosen by the parent model).
    This block only handles the time mixing part.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 use_ternary: bool = True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.time_mixing = RWKV7TimeMixing(d_model, n_heads, d_head, use_ternary)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        prev_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            state: [batch, n_heads, d_head, d_head]
            prev_x: [batch, 1, d_model]
        Returns:
            x + time_mixing_output: [batch, seq_len, d_model]
            new_state
            last_x
        """
        residual = x
        x_norm = self.norm(x)
        tm_out, new_state, last_x = self.time_mixing(x_norm, state, prev_x)
        return residual + tm_out, new_state, last_x
