"""
model.py - a small GPT (decoder-only Transformer) written from scratch.

The whole idea: given a sequence of tokens, predict the next token.
  tokens -> embeddings -> N x [attention + MLP] -> scores for every possible next token
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256    # context length: how many tokens the model can see at once
    vocab_size: int = 65     # number of different tokens
    n_layer: int = 6         # number of Transformer blocks stacked on top of each other
    n_head: int = 6          # attention heads per block
    n_embd: int = 384        # size of each token's vector
    dropout: float = 0.2     # randomly zero some activations during training (prevents memorising)
    bias: bool = False       # use bias in Linear/LayerNorm layers (False is slightly better and faster)


class CausalSelfAttention(nn.Module):
    """Every token looks back at earlier tokens and decides which ones matter."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # One Linear makes query, key and value for all heads at once.
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape                       # batch, time (sequence length), channels
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        hs = C // self.n_head                   # head size
        # (B, T, C) -> (B, n_head, T, head_size) so each head works separately
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # softmax(q @ k^T / sqrt(hs)) @ v, with a causal mask so tokens can't see the future.
        # PyTorch's fused version (Flash Attention on the GPU) is much faster and lighter on VRAM.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # glue the heads back together
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """A small feed-forward network applied to each token on its own: 'thinking' after attention."""

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    """One Transformer block: LayerNorm -> attention -> add, LayerNorm -> MLP -> add."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection: keep the original, add new info
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: embeddings + stacked blocks + output layer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # what the token is
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # where the token is
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the input embedding and output layer share one matrix (fewer params).
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the init of layers that write into the residual stream,
        # so the signal doesn't grow as more blocks are stacked.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence {T} longer than block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # Training: score every position and compare with the real next token.
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        else:
            # Generating: only the last position's prediction is needed (saves work).
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """AdamW. Weight decay only on matrices (2D), not on biases / LayerNorm weights."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """
        Yield new tokens one at a time (so text can appear live on screen).
        idx: (B, T) tensor of starting token ids. Call model.eval() first.
        """
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens, so crop to the most recent ones.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)    # always the top choice
            else:
                logits = logits / temperature          # <1 = safer, >1 = wilder
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")    # only keep the top_k options
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)   # roll the dice
            idx = torch.cat((idx, next_id), dim=1)
            yield next_id
            if stop_token is not None and bool((next_id == stop_token).all()):
                break

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        """Generate all at once. Returns the prompt plus the new tokens, shape (B, T + new)."""
        out = [idx]
        for next_id in self.generate_stream(idx, max_new_tokens, temperature, top_k, stop_token):
            out.append(next_id)
        return torch.cat(out, dim=1)
