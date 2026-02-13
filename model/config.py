"""
Aria Model Configurations.
Defines Tiny/Small/Full model hyperparameters.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AriaConfig:
    # Model dimensions
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    n_kv_heads: int = 4              # GQA: fewer KV heads than Q heads

    # MoE
    n_shared_experts: int = 1
    n_routed_experts: int = 2
    d_expert: int = 512              # inner dim per expert FFN
    top_k_experts: int = 1           # top-1 routing

    # Vocabulary
    vocab_size: int = 32768

    # Sequence
    max_seq_len: int = 2048
    attention_window: int = 512      # sliding window for attention layers

    # Layer pattern: RWKV-RWKV-RWKV-Attn repeating
    rwkv_to_attn_ratio: int = 3
    # Shared attention: all attention layers share one weight set
    share_attention_weights: bool = True

    # RWKV-7 specific
    rwkv_state_size: int = 64        # state matrix size per head (d_head × d_head)

    # BitLinear
    use_ternary: bool = True         # False for FP baseline comparison

    # Speculative decoding
    use_draft_head: bool = True
    draft_head_layer: int = -1       # auto-computed: 2nd attention checkpoint
    draft_tokens: int = 3

    # Training
    tie_embeddings: bool = True
    dropout: float = 0.0             # no dropout for ternary training
    init_std: float = 0.02

    # Precision
    dtype: str = "bfloat16"

    def __post_init__(self):
        assert self.d_model == self.n_heads * self.d_head, \
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * d_head ({self.d_head})"
        assert self.d_model % self.n_kv_heads == 0, \
            f"d_model must be divisible by n_kv_heads"
        assert self.n_layers % (self.rwkv_to_attn_ratio + 1) == 0, \
            f"n_layers ({self.n_layers}) must be divisible by (rwkv_ratio + 1) = {self.rwkv_to_attn_ratio + 1}"

        # Auto-compute draft head placement at 2nd attention checkpoint
        if self.draft_head_layer < 0:
            layers_per_group = self.rwkv_to_attn_ratio + 1
            self.draft_head_layer = 2 * layers_per_group - 1  # end of 2nd group

    @property
    def n_rwkv_layers(self) -> int:
        groups = self.n_layers // (self.rwkv_to_attn_ratio + 1)
        return groups * self.rwkv_to_attn_ratio

    @property
    def n_attn_layers(self) -> int:
        return self.n_layers // (self.rwkv_to_attn_ratio + 1)

    @property
    def layer_types(self) -> list[Literal["rwkv", "attn"]]:
        """Returns ordered list of layer types: ['rwkv','rwkv','rwkv','attn', ...]"""
        pattern = ["rwkv"] * self.rwkv_to_attn_ratio + ["attn"]
        return pattern * (self.n_layers // len(pattern))

    def total_params_estimate(self) -> dict[str, int]:
        """Rough parameter count breakdown."""
        # Embedding (tied → count once)
        embed = self.vocab_size * self.d_model

        # RWKV layers: R,K,V,G,O projections
        rwkv_time = self.n_rwkv_layers * 5 * self.d_model * self.d_model

        # MoE FFN per RWKV layer
        expert_ffn = 3 * self.d_model * self.d_expert  # SwiGLU: up + gate + down
        moe_per_layer = (self.n_shared_experts + self.n_routed_experts) * expert_ffn
        moe_router = self.d_model * self.n_routed_experts
        rwkv_moe = self.n_rwkv_layers * (moe_per_layer + moe_router)

        # Shared attention (1 weight set for all attn layers)
        # Q projection
        attn_q = self.d_model * self.d_model
        # K,V projections (GQA: fewer heads)
        kv_dim = self.n_kv_heads * self.d_head
        attn_kv = 2 * self.d_model * kv_dim
        # O projection
        attn_o = self.d_model * self.d_model
        attn_weights = attn_q + attn_kv + attn_o
        # Attention FFN (one shared set, same SwiGLU pattern)
        attn_ffn = 3 * self.d_model * self.d_expert * (self.n_shared_experts + self.n_routed_experts)
        shared_attn_total = attn_weights + attn_ffn

        # Norms: 2 per layer × d_model
        norms = self.n_layers * 2 * self.d_model

        # RWKV misc: decay, eta per layer per head
        rwkv_misc = self.n_rwkv_layers * self.n_heads * 2

        # Draft head
        draft = 3 * self.d_model * self.d_model if self.use_draft_head else 0

        total = embed + rwkv_time + rwkv_moe + shared_attn_total + norms + rwkv_misc + draft

        return {
            "embedding": embed,
            "rwkv_time_mixing": rwkv_time,
            "rwkv_moe_ffn": rwkv_moe,
            "shared_attention": shared_attn_total,
            "norms": norms,
            "rwkv_misc": rwkv_misc,
            "draft_head": draft,
            "total": total,
        }


# ─── Preset Configs ─────────────────────────────────────────

def aria_tiny() -> AriaConfig:
    """~200M params. For proof-of-concept on Colab."""
    return AriaConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_head=64,
        n_kv_heads=4,
        n_shared_experts=1,
        n_routed_experts=2,
        d_expert=512,
        top_k_experts=1,
        vocab_size=32768,
    )


def aria_small() -> AriaConfig:
    """~520M params. For Phase 2 training."""
    return AriaConfig(
        d_model=1280,
        n_layers=20,
        n_heads=20,
        d_head=64,
        n_kv_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        d_expert=1024,
        top_k_experts=1,
        vocab_size=32768,
    )


def aria_full() -> AriaConfig:
    """~1.8B params. Full model for H100 training."""
    return AriaConfig(
        d_model=2048,
        n_layers=24,
        n_heads=32,
        d_head=64,
        n_kv_heads=4,
        n_shared_experts=1,
        n_routed_experts=6,
        d_expert=2048,
        top_k_experts=1,
        vocab_size=32768,
    )


def aria_debug() -> AriaConfig:
    """Tiny config for unit tests. Runs on CPU in seconds."""
    return AriaConfig(
        d_model=64,
        n_layers=4,
        n_heads=4,
        d_head=16,
        n_kv_heads=2,
        n_shared_experts=1,
        n_routed_experts=1,
        d_expert=32,
        top_k_experts=1,
        vocab_size=256,
        max_seq_len=128,
        attention_window=32,
        use_draft_head=False,
    )


if __name__ == "__main__":
    for name, fn in [("debug", aria_debug), ("tiny", aria_tiny),
                     ("small", aria_small), ("full", aria_full)]:
        cfg = fn()
        params = cfg.total_params_estimate()
        total_m = params["total"] / 1e6
        print(f"\n{'='*50}")
        print(f"Aria-{name.upper()}: {total_m:.1f}M params")
        print(f"  Layers: {cfg.n_layers} ({cfg.n_rwkv_layers}R + {cfg.n_attn_layers}A)")
        print(f"  Layer pattern: {cfg.layer_types}")
        print(f"  d_model={cfg.d_model}, n_heads={cfg.n_heads}, d_head={cfg.d_head}")
        print(f"  MoE: {cfg.n_shared_experts}s + {cfg.n_routed_experts}r, d_expert={cfg.d_expert}")
        for k, v in params.items():
            if k != "total":
                print(f"    {k}: {v/1e6:.1f}M")
        ternary_mb = (params["total"] - params["embedding"]) * 2 / 8 / 1e6
        embed_mb = params["embedding"] / 1e6  # INT8
        print(f"  Disk (ternary+INT8 embed): {ternary_mb + embed_mb:.0f} MB")
