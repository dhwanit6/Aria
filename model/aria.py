"""
Aria — Full Model Assembly.

Architecture (24-layer example):
    Embedding → [RWKV, RWKV, RWKV, SharedAttn] × 6 → RMSNorm → LM Head

Each layer consists of:
    Time/Attention Mixing → Channel Mixing (MoE FFN)

Key design:
- RWKV-7 layers handle sequential state (long-range, O(1) inference)
- Shared Attention layers provide associative recall (short-range, O(W) window)
- All attention layers share ONE set of weights (Zamba trick)
- BitLinear ternary weights throughout (except router, norms, embedding)
- MoE FFN on every layer (shared + routed experts)
- Draft head at 2nd attention checkpoint for speculative decoding
- Tied embedding (input and output share weights)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .config import AriaConfig
from .rmsnorm import RMSNorm
from .rwkv7_block import RWKV7Block
from .shared_attention import SharedAttentionBlock
from .moe_ffn import ChannelMixingBlock
from .draft_head import DraftHead


@dataclass
class AriaOutput:
    """Model output container."""
    logits: torch.Tensor                              # [B, T, vocab]
    loss: torch.Tensor | None = None                  # scalar
    draft_logits: list[torch.Tensor] | None = None    # list of [B, T, vocab]
    draft_loss: torch.Tensor | None = None            # scalar
    aux_loss: torch.Tensor | None = None              # MoE balance loss
    rwkv_states: list[torch.Tensor] | None = None     # per-layer RWKV states
    kv_caches: list | None = None                     # per-attn-layer KV caches
    prev_xs: list[torch.Tensor] | None = None         # per-RWKV-layer last x


class Aria(nn.Module):
    """
    The Aria hybrid language model.
    
    RWKV-7 backbone with shared sliding-window attention checkpoints,
    BitNet ternary weights, MoE channel mixing, and speculative draft head.
    """

    def __init__(self, config: AriaConfig):
        super().__init__()
        self.config = config

        # ──── Embedding ────
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=config.init_std)

        # ──── Build layer stack ────
        self.layers_time_mix = nn.ModuleList()   # RWKV or Attention blocks
        self.layers_chan_mix = nn.ModuleList()    # MoE FFN blocks (one per layer)

        # Shared attention: create ONE attention block, reference it multiple times
        shared_attn = SharedAttentionBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            d_head=config.d_head,
            window_size=config.attention_window,
            max_seq_len=config.max_seq_len,
            use_ternary=config.use_ternary,
        )

        # Track which layers are RWKV vs attention (for state management)
        self.layer_types = config.layer_types
        self._rwkv_layer_indices: list[int] = []
        self._attn_layer_indices: list[int] = []

        for i, ltype in enumerate(self.layer_types):
            if ltype == "rwkv":
                self.layers_time_mix.append(
                    RWKV7Block(
                        d_model=config.d_model,
                        n_heads=config.n_heads,
                        d_head=config.d_head,
                        use_ternary=config.use_ternary,
                    )
                )
                self._rwkv_layer_indices.append(i)
            else:  # "attn"
                if config.share_attention_weights:
                    # All attention layers share the SAME block
                    self.layers_time_mix.append(shared_attn)
                else:
                    # Independent attention weights per layer
                    self.layers_time_mix.append(
                        SharedAttentionBlock(
                            d_model=config.d_model,
                            n_heads=config.n_heads,
                            n_kv_heads=config.n_kv_heads,
                            d_head=config.d_head,
                            window_size=config.attention_window,
                            max_seq_len=config.max_seq_len,
                            use_ternary=config.use_ternary,
                        )
                    )
                self._attn_layer_indices.append(i)

            # Channel mixing (MoE FFN) — each layer gets its own
            self.layers_chan_mix.append(
                ChannelMixingBlock(
                    d_model=config.d_model,
                    d_expert=config.d_expert,
                    n_shared_experts=config.n_shared_experts,
                    n_routed_experts=config.n_routed_experts,
                    top_k=config.top_k_experts,
                    use_moe=config.n_routed_experts > 0,
                    use_ternary=config.use_ternary,
                )
            )

        # ──── Final norm + LM head ────
        self.final_norm = RMSNorm(config.d_model)

        # LM head: tied with embedding (no extra params)
        # We'll use F.linear(h, self.embedding.weight) in forward()

        # ──── Draft head (optional) ────
        self.draft_head: DraftHead | None = None
        if config.use_draft_head:
            self.draft_head = DraftHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                n_draft_tokens=config.draft_tokens,
                use_ternary=config.use_ternary,
            )

        # ──── Gradient checkpointing flag ────
        self.gradient_checkpointing = False
        self.checkpoint_use_reentrant = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def set_gradient_checkpointing(self, enable: bool = True):
        """Enable gradient checkpointing to trade compute for memory."""
        self.gradient_checkpointing = enable

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        rwkv_states: list[torch.Tensor] | None = None,
        kv_caches: list | None = None,
        prev_xs: list[torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> AriaOutput:
        """
        Full forward pass.

        Args:
            input_ids: [batch, seq_len] — token indices
            target_ids: [batch, seq_len] — next-token targets for loss (optional)
            rwkv_states: list of [B, H, D, D] RWKV states (one per RWKV layer)
            kv_caches: list of (k, v) caches (one per attention layer)
            prev_xs: list of [B, 1, C] previous x for token shift (one per RWKV layer)
            position_offset: position offset for RoPE (for generation mode)

        Returns:
            AriaOutput with logits, losses, and updated states/caches
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ──── Embedding ────
        h = self.embedding(input_ids)  # [B, T, d_model]

        # ──── Initialize states if needed ────
        if rwkv_states is None:
            rwkv_states = [None] * len(self._rwkv_layer_indices)
        if kv_caches is None:
            kv_caches = [None] * len(self._attn_layer_indices)
        if prev_xs is None:
            prev_xs = [None] * len(self._rwkv_layer_indices)

        # ──── Layer stack ────
        new_rwkv_states = []
        new_kv_caches = []
        new_prev_xs = []
        rwkv_idx = 0
        attn_idx = 0

        total_aux_loss = torch.tensor(0.0, device=device)
        has_aux_loss = False
        draft_logits = None

        for i, (time_block, chan_block) in enumerate(
            zip(self.layers_time_mix, self.layers_chan_mix)
        ):
            # ──── Time mixing ────
            if self.layer_types[i] == "rwkv":
                if self.gradient_checkpointing and self.training:
                    h, new_state, last_x = torch.utils.checkpoint.checkpoint(
                        time_block, h, rwkv_states[rwkv_idx], prev_xs[rwkv_idx],
                        use_reentrant=self.checkpoint_use_reentrant,
                    )
                else:
                    h, new_state, last_x = time_block(
                        h, rwkv_states[rwkv_idx], prev_xs[rwkv_idx]
                    )
                new_rwkv_states.append(new_state)
                new_prev_xs.append(last_x)
                rwkv_idx += 1
            else:  # attention
                if self.gradient_checkpointing and self.training:
                    h, new_kv = torch.utils.checkpoint.checkpoint(
                        time_block, h, kv_caches[attn_idx], position_offset,
                        use_reentrant=self.checkpoint_use_reentrant,
                    )
                else:
                    h, new_kv = time_block(h, kv_caches[attn_idx], position_offset)
                new_kv_caches.append(new_kv)
                attn_idx += 1

            # ──── Channel mixing (MoE FFN) ────
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    chan_block, h, use_reentrant=self.checkpoint_use_reentrant,
                )
            else:
                h = chan_block(h)

            # Collect MoE auxiliary loss
            if chan_block.aux_loss is not None:
                total_aux_loss = total_aux_loss + chan_block.aux_loss
                has_aux_loss = True

            # ──── Draft head (at the designated layer) ────
            if (
                self.draft_head is not None
                and i == self.config.draft_head_layer
                and self.training
            ):
                draft_logits = self.draft_head(h, self.embedding.weight)

        # ──── Final norm + LM head ────
        h = self.final_norm(h)
        logits = F.linear(h, self.embedding.weight)  # [B, T, vocab] — tied weights

        # ──── Compute losses ────
        loss = None
        draft_loss = None

        if target_ids is not None:
            # Main language modeling loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
            )

            # Draft head loss (multi-task training signal)
            if draft_logits is not None:
                draft_loss = self.draft_head.compute_loss(
                    draft_logits, target_ids, loss_weight=0.3
                )

            # Add auxiliary losses
            if has_aux_loss:
                loss = loss + total_aux_loss

            if draft_loss is not None:
                loss = loss + draft_loss

        return AriaOutput(
            logits=logits,
            loss=loss,
            draft_logits=draft_logits,
            draft_loss=draft_loss,
            aux_loss=total_aux_loss if has_aux_loss else None,
            rwkv_states=new_rwkv_states,
            kv_caches=new_kv_caches,
            prev_xs=new_prev_xs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with nucleus sampling.
        
        Uses RWKV state + KV cache for O(1) per-token inference.
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        # Prefill: process the entire prompt
        output = self.forward(input_ids)
        rwkv_states = output.rwkv_states
        kv_caches = output.kv_caches
        prev_xs = output.prev_xs
        position = input_ids.shape[1]

        # Get the last token's logits
        next_logits = output.logits[:, -1, :]  # [B, vocab]

        generated = []

        for _ in range(max_new_tokens):
            # ──── Sample ────
            logits = next_logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                all_tokens = torch.cat(
                    [input_ids] + ([torch.stack(generated, dim=1)] if generated else []),
                    dim=1,
                )
                for b in range(B):
                    seen = all_tokens[b].unique()
                    logits[b, seen] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = logits.topk(top_k, dim=-1)
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative prob above threshold
                mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                # Scatter back
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated.append(next_token.squeeze(-1))

            # ──── Forward single token with cached states ────
            output = self.forward(
                next_token,
                rwkv_states=rwkv_states,
                kv_caches=kv_caches,
                prev_xs=prev_xs,
                position_offset=position,
            )
            rwkv_states = output.rwkv_states
            kv_caches = output.kv_caches
            prev_xs = output.prev_xs
            next_logits = output.logits[:, -1, :]
            position += 1

        return torch.stack(generated, dim=1)  # [B, max_new_tokens]

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts: dict[str, int] = {}

        counts["embedding"] = sum(p.numel() for p in self.embedding.parameters())

        rwkv_params = 0
        attn_params = 0
        moe_params = 0
        seen_attn_ids: set[int] = set()

        for i, (tm, cm) in enumerate(zip(self.layers_time_mix, self.layers_chan_mix)):
            tm_count = sum(p.numel() for p in tm.parameters())
            cm_count = sum(p.numel() for p in cm.parameters())

            if self.layer_types[i] == "rwkv":
                rwkv_params += tm_count
            else:
                # Only count shared attention once
                obj_id = id(tm)
                if obj_id not in seen_attn_ids:
                    attn_params += tm_count
                    seen_attn_ids.add(obj_id)

            moe_params += cm_count

        counts["rwkv_time_mixing"] = rwkv_params
        counts["shared_attention"] = attn_params
        counts["moe_ffn"] = moe_params
        counts["final_norm"] = sum(p.numel() for p in self.final_norm.parameters())

        if self.draft_head is not None:
            counts["draft_head"] = sum(p.numel() for p in self.draft_head.parameters())

        counts["total"] = sum(counts.values())

        # Also count UNIQUE parameters (shared attention counted once)
        seen_ids: set[int] = set()
        unique_total = 0
        for p in self.parameters():
            pid = id(p)
            if pid not in seen_ids:
                unique_total += p.numel()
                seen_ids.add(pid)
        counts["total_unique"] = unique_total

        return counts


def build_aria(config: AriaConfig) -> Aria:
    """Build an Aria model from config."""
    model = Aria(config)

    # Print parameter summary
    counts = model.count_parameters()
    total_m = counts["total_unique"] / 1e6
    print(f"Aria model built: {total_m:.1f}M unique parameters")
    for k, v in counts.items():
        if k not in ("total", "total_unique"):
            print(f"  {k}: {v/1e6:.1f}M")
    print(f"  Layer pattern: {'→'.join(model.layer_types)}")

    return model
