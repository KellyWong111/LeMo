"""
Exact-kernel FDA predictor for LeWM.

This variant follows the public FDA kernel as closely as possible:

- keep LeWM predictor structure
- keep learnable positional embedding
- keep AdaLN-zero conditional blocks
- replace predictor attention with the official FDA Triton kernel
"""

import os
import sys

import torch
import torch.nn as nn
from einops import rearrange

_MODA_ROOT_CANDIDATES = [
    os.environ.get("MODA_TRITON_ROOT"),
    "/data1/jingyixi/.cache_runtime/MoDA/libs/moda_triton",
    "/home/internship/wm_transfer_lab/MoDA/libs/moda_triton",
]
for _moda_root in _MODA_ROOT_CANDIDATES:
    if _moda_root and _moda_root not in sys.path:
        sys.path.insert(0, _moda_root)

try:
    from fla.ops.moda.fda_v12 import parallel_fda

    FDA_TRITON_AVAILABLE = True
except ImportError:
    FDA_TRITON_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FDAAttentionExact(nn.Module):
    """LeWM attention layout with exact FDA kernel in the predictor."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cached_k=None, cached_v=None):
        normed = self.norm(x)
        qkv = self.to_qkv(normed).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b t h d", h=self.heads) for t in qkv)

        cur_k = k
        cur_v = v

        if cached_k is None or cached_v is None:
            out = torch.zeros_like(v)
        else:
            out = parallel_fda(
                q=q,
                k=None,
                v=None,
                cached_k=cached_k,
                cached_v=cached_v,
                scale=self.scale,
                dsa_group_num=1,
                head_first=False,
                need_lse=False,
                warn_shape=False,
            )

        out = rearrange(out, "b t h d -> b t (h d)")
        return self.to_out(out), cur_k, cur_v


class FDAConditionalBlockExact(nn.Module):
    """LeWM ConditionalBlock with FDA-attention drop-in."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = FDAAttentionExact(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, cached_k=None, cached_v=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, cur_k, cur_v = self.attn(attn_input, cached_k, cached_v)
        x = x + gate_msa * attn_out
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, cur_k, cur_v


class FDATransformerExact(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [
                FDAConditionalBlockExact(
                    hidden_dim, heads, dim_head, mlp_dim, dropout
                )
                for _ in range(depth)
            ]
        )

    @staticmethod
    def _build_depth_cache(k_list, v_list, t):
        if not k_list:
            return None, None
        stacked_k = torch.stack(k_list, dim=2)
        stacked_v = torch.stack(v_list, dim=2)
        b, _t, l, h, d = stacked_k.shape
        return (
            stacked_k.reshape(b, t * l, h, d),
            stacked_v.reshape(b, t * l, h, stacked_v.shape[-1]),
        )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        t = x.shape[1]
        k_cache_list = []
        v_cache_list = []

        for block in self.layers:
            if k_cache_list:
                cached_k, cached_v = self._build_depth_cache(k_cache_list, v_cache_list, t)
            else:
                cached_k, cached_v = None, None

            x, cur_k, cur_v = block(x, c, cached_k, cached_v)
            k_cache_list.append(cur_k)
            v_cache_list.append(cur_v)

        x = self.norm(x)
        return self.output_proj(x)


class FDAARPredictorExact(nn.Module):
    """Historical strong-path predictor: LeWM skeleton + FDA kernel."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = FDATransformerExact(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x, c):
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)
