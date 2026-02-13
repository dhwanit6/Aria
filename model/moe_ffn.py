"""
Mixture-of-Experts FFN with shared + routed experts.

Architecture:
    1 shared expert (always active) + N routed experts (top-k selected)
    Each expert is a SwiGLU FFN: SiLU(W_up · x) ⊙ (W_gate · x) → W_down

Design choices:
- Top-1 routing for simplicity and speed (only 2 FFN evaluations per token)
- Load balancing loss to prevent expert collapse (all tokens going to one expert)
- Shared expert handles common patterns, routed experts handle domain specialization
- No auxiliary "jitter" noise — clean routing via learned gate

Reference: DeepSeek-MoE (Dai et al., 2024), Switch Transformer (Fedus et al., 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import LinearFlex
from .rmsnorm import RMSNorm


class SwiGLUExpert(nn.Module):
    """
    Single SwiGLU FFN expert.
    
    SwiGLU(x) = SiLU(W_up · x) ⊙ (W_gate · x)
    out = W_down · SwiGLU(x)
    
    The gated structure (up × gate) provides better gradient flow than
    standard FFN, especially important with ternary weights where every
    bit of expressivity counts.
    """

    def __init__(self, d_model: int, d_expert: int, use_ternary: bool = True):
        super().__init__()
        self.W_up = LinearFlex(d_model, d_expert, use_ternary=use_ternary)
        self.W_gate = LinearFlex(d_model, d_expert, use_ternary=use_ternary)
        self.W_down = LinearFlex(d_expert, d_model, use_ternary=use_ternary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_up(x)) * self.W_gate(x))


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts Feed-Forward Network.
    
    Forward path per token:
        1. Shared expert output (always computed)
        2. Router selects top-k routed experts
        3. Selected expert outputs weighted by router scores
        4. Final = shared_out + weighted_routed_out
    
    Load balancing:
        We add an auxiliary loss that encourages uniform expert selection.
        Without this, the router can collapse to always picking one expert,
        wasting the other experts' parameters.
        
        L_balance = α · N · Σᵢ (fᵢ · pᵢ)
        where fᵢ = fraction of tokens routed to expert i
              pᵢ = mean router probability for expert i
              N = number of experts
              α = balance loss weight (default 0.01)
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_shared_experts: int = 1,
        n_routed_experts: int = 4,
        top_k: int = 1,
        balance_loss_weight: float = 0.01,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_shared = n_shared_experts
        self.n_routed = n_routed_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_expert, use_ternary) for _ in range(n_shared_experts)
        ])

        # Routed experts (top-k selected per token)
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(d_model, d_expert, use_ternary) for _ in range(n_routed_experts)
        ])

        # Router: projects hidden state to expert scores
        # NOT ternary — router needs full precision for clean routing decisions
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

        # Track auxiliary loss for training
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        Side effect:
            self.aux_loss is set to the load balance loss for this forward pass.
        """
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)  # [B*T, C]
        N = x_flat.shape[0]  # total tokens

        # ──── Shared expert(s) ────
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # ──── Router ────
        router_logits = self.router(x_flat.detach() if not self.training else x_flat)
        router_probs = F.softmax(router_logits, dim=-1)  # [N, n_routed]

        # Top-k expert selection
        top_k_vals, top_k_indices = router_probs.topk(self.top_k, dim=-1)  # [N, k]

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # ──── Routed expert computation ────
        # For top-1 routing: just one expert per token
        routed_out = torch.zeros_like(x_flat)  # [N, C]

        if self.top_k == 1:
            # Optimized path for top-1: group tokens by expert, batch compute
            expert_idx = top_k_indices.squeeze(-1)  # [N]
            for i, expert in enumerate(self.routed_experts):
                mask = expert_idx == i
                if mask.any():
                    routed_out[mask] = expert(x_flat[mask]) * top_k_weights[mask]
        else:
            # General top-k path
            for k_idx in range(self.top_k):
                expert_indices = top_k_indices[:, k_idx]  # [N]
                weights = top_k_weights[:, k_idx].unsqueeze(-1)  # [N, 1]
                for i, expert in enumerate(self.routed_experts):
                    mask = expert_indices == i
                    if mask.any():
                        routed_out[mask] += expert(x_flat[mask]) * weights[mask]

        # ──── Combine ────
        output = shared_out + routed_out
        output = output.reshape(B, T, C)

        # ──── Load balance auxiliary loss ────
        if self.training:
            self._compute_balance_loss(router_probs, top_k_indices, N)

        return output

    def _compute_balance_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        n_tokens: int,
    ) -> None:
        """
        Compute load balancing loss to prevent expert collapse.
        
        L = α · n_experts · Σᵢ (fᵢ · pᵢ)
        
        fᵢ = fraction of tokens assigned to expert i
        pᵢ = average router probability for expert i
        
        This loss is MINIMIZED when all experts receive equal traffic
        AND equal probability. The product fᵢ·pᵢ penalizes correlation
        between "chosen often" and "high probability" — discouraging
        winner-take-all collapse.
        """
        # Fraction of tokens routed to each expert
        # Count how many times each expert appears in top-k selections
        counts = torch.zeros(self.n_routed, device=router_probs.device)
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices[:, k_idx]
            for i in range(self.n_routed):
                counts[i] += (expert_indices == i).float().sum()
        f = counts / (n_tokens * self.top_k)  # [n_routed]

        # Mean router probability per expert
        p = router_probs.mean(dim=0)  # [n_routed]

        # Balance loss
        self.aux_loss = self.balance_loss_weight * self.n_routed * (f * p).sum()


import math  # needed for kaiming init


class DenseFFN(nn.Module):
    """
    Dense SwiGLU FFN (no MoE). Used for ablation / comparison.
    Also used as the FFN in attention layers if different from RWKV layers.
    """

    def __init__(self, d_model: int, d_ffn: int, use_ternary: bool = True):
        super().__init__()
        self.ffn = SwiGLUExpert(d_model, d_ffn, use_ternary)
        self.aux_loss = None  # Interface compatibility with MoEFFN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class ChannelMixingBlock(nn.Module):
    """
    Channel mixing: RMSNorm → MoE/Dense FFN → Residual.
    This is the second half of each layer (after time mixing).
    """

    def __init__(
        self,
        d_model: int,
        d_expert: int,
        n_shared_experts: int = 1,
        n_routed_experts: int = 4,
        top_k: int = 1,
        use_moe: bool = True,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)

        if use_moe and n_routed_experts > 0:
            self.ffn = MoEFFN(
                d_model=d_model,
                d_expert=d_expert,
                n_shared_experts=n_shared_experts,
                n_routed_experts=n_routed_experts,
                top_k=top_k,
                use_ternary=use_ternary,
            )
        else:
            self.ffn = DenseFFN(d_model, d_expert, use_ternary)

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return self.ffn.aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))
