"""
Speculative Decoding Draft Head.

Attached to an early layer of the model (2nd attention checkpoint).
Predicts the next 1-3 tokens in parallel to enable speculative decoding.

During inference:
    1. DraftHead predicts 3 draft tokens from early layer output
    2. Main model continues processing all layers for the "real" token
    3. On next step, we verify drafts by running them through the full model in batch
    4. Accept matching prefix → effective 1.5-2x throughput for predictable text

During training:
    Multi-task loss: main LM head loss + draft head loss (weighted lower).
    This is "deep supervision" — forces intermediate representations to be predictive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import LinearFlex
from .rmsnorm import RMSNorm


class DraftHead(nn.Module):
    """
    Lightweight draft token predictor.
    
    Architecture:
        h_early → RMSNorm → Linear(d_model, d_model) → SiLU → Linear(d_model, vocab)
    
    For multi-token prediction, we use an autoregressive chain:
        draft_1 = argmax(head(h))
        draft_2 = argmax(head(h + embed(draft_1)))
        draft_3 = argmax(head(h + embed(draft_2)))
    
    The key insight: we REUSE the tied embedding matrix as the output projection.
    This costs almost zero extra parameters (just the intermediate linear + norm).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_draft_tokens: int = 3,
        use_ternary: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_draft_tokens = n_draft_tokens

        self.norm = RMSNorm(d_model)
        # Projection that maps early features to output-space features
        self.proj = LinearFlex(d_model, d_model, use_ternary=use_ternary)
        # Residual projections for chained prediction (one per additional draft token)
        self.chain_projs = nn.ModuleList([
            LinearFlex(d_model, d_model, use_ternary=use_ternary)
            for _ in range(n_draft_tokens - 1)
        ])

    def forward(
        self,
        h: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Predict multiple draft tokens from early-layer hidden state.

        Args:
            h: [batch, seq_len, d_model] — output of early layer
            embedding_weight: [vocab_size, d_model] — tied embedding matrix

        Returns:
            logits_list: list of n_draft_tokens tensors, each [batch, seq_len, vocab_size]
        """
        h = self.norm(h)
        h_proj = F.silu(self.proj(h))  # [B, T, d_model]

        logits_list = []

        # First draft token: project to vocab via tied embedding
        logits_1 = F.linear(h_proj, embedding_weight)  # [B, T, vocab]
        logits_list.append(logits_1)

        # Chained predictions for draft tokens 2..n
        h_chain = h_proj
        for i, chain_proj in enumerate(self.chain_projs):
            # During training: use soft embedding (differentiable)
            # softmax over logits → weighted embedding lookup
            if self.training:
                soft_probs = F.softmax(logits_list[-1].detach(), dim=-1)
                soft_embed = F.linear(soft_probs, embedding_weight.T)  # [B, T, d_model]
            else:
                # During inference: use hard argmax (non-differentiable)
                token_ids = logits_list[-1].argmax(dim=-1)  # [B, T]
                soft_embed = F.embedding(token_ids, embedding_weight)  # [B, T, d_model]

            # Residual update: incorporate the predicted token info
            h_chain = h_chain + F.silu(chain_proj(soft_embed))
            logits_n = F.linear(h_chain, embedding_weight)
            logits_list.append(logits_n)

        return logits_list

    def compute_loss(
        self,
        draft_logits: list[torch.Tensor],
        target_ids: torch.Tensor,
        loss_weight: float = 0.3,
    ) -> torch.Tensor:
        """
        Multi-token prediction loss for training.
        
        For each draft position i, the target is the token at position (t+i+1).
        We weight later predictions lower since they're harder and less reliable.

        Args:
            draft_logits: list of [B, T, vocab] from forward()
            target_ids: [B, T] — the standard next-token targets
            loss_weight: overall weight relative to main LM loss

        Returns:
            Weighted sum of cross-entropy losses across draft positions.
        """
        total_loss = torch.tensor(0.0, device=target_ids.device)
        B, T = target_ids.shape

        for i, logits in enumerate(draft_logits):
            # Draft position i predicts token at offset (i+1)
            # So target for draft_i at position t is target_ids at position (t+i)
            # We need to shift: logits[:, :T-i-1] predicts target_ids[:, i+1:T]
            # But since target_ids is already shifted (it's next-token), draft_i
            # predicts target_ids[:, i:]
            offset = i
            if offset >= T:
                break

            shifted_logits = logits[:, : T - offset, :]  # [B, T-offset, vocab]
            shifted_targets = target_ids[:, offset:]  # [B, T-offset]

            loss_i = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_targets.reshape(-1),
                ignore_index=-100,
            )

            # Decay weight for later predictions (they're inherently less accurate)
            position_weight = 1.0 / (i + 1)
            total_loss = total_loss + loss_weight * position_weight * loss_i

        return total_loss
