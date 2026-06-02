"""
Exact-kernel MoDA predictor for LeWM.

This variant stays close to the historical strong path:

- keep LeWM predictor structure
- keep learnable positional embedding
- keep AdaLN-zero conditional blocks
- replace predictor attention with the official MoDA Triton kernel
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    from fla.ops.moda import parallel_moda, parallel_moda_chunk_visible

    MODA_TRITON_AVAILABLE = True
except ImportError:
    MODA_TRITON_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MoDAAttentionExact(nn.Module):
    """LeWM attention layout with exact MoDA kernel in the predictor."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        chunk_visible=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.chunk_visible = bool(chunk_visible)

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

        has_depth = (cached_k is not None) and (cached_v is not None)
        if has_depth:
            moda_kernel = (
                parallel_moda_chunk_visible
                if bool(getattr(self, "chunk_visible", False))
                else parallel_moda
            )
            out = moda_kernel(
                q,
                k,
                v,
                cached_k=cached_k,
                cached_v=cached_v,
                scale=self.scale,
                moda_group_num=1,
                head_first=False,
                need_lse=False,
                warn_shape=False,
            )
            out = rearrange(out, "b t h d -> b t (h d)")
            return self.to_out(out), cur_k, cur_v

        q_hf = rearrange(q, "b t h d -> b h t d")
        k_hf = rearrange(k, "b t h d -> b h t d")
        v_hf = rearrange(v, "b t h d -> b h t d")
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_hf, k_hf, v_hf, dropout_p=drop, is_causal=True
        )
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out), cur_k, cur_v


class MoDAConditionalBlockExact(nn.Module):
    """LeWM ConditionalBlock with MoDA-attention drop-in."""

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        chunk_visible=False,
    ):
        super().__init__()
        self.attn = MoDAAttentionExact(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            chunk_visible=chunk_visible,
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


class MoDATransformerExact(nn.Module):
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
        use_moda=True,
        depth_start_layer=1,
        cache_window=0,
        detach_cache=False,
        chunk_visible=False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depth = depth
        self.use_moda = use_moda
        self.depth_start_layer = depth_start_layer
        self.cache_window = int(cache_window)
        self.detach_cache = bool(detach_cache)
        self.chunk_visible = bool(chunk_visible)

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
                MoDAConditionalBlockExact(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                    chunk_visible=chunk_visible,
                )
                for _ in range(depth)
            ]
        )

    def _build_depth_cache(self, k_list, v_list, t):
        if not k_list:
            return None, None
        cache_window = int(getattr(self, "cache_window", 0))
        if cache_window > 0:
            k_list = k_list[-cache_window:]
            v_list = v_list[-cache_window:]
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

        for layer_idx, block in enumerate(self.layers):
            if self.use_moda and layer_idx >= self.depth_start_layer and k_cache_list:
                cached_k, cached_v = self._build_depth_cache(k_cache_list, v_cache_list, t)
            else:
                cached_k, cached_v = None, None

            x, cur_k, cur_v = block(x, c, cached_k, cached_v)
            if getattr(self, "detach_cache", False):
                cur_k = cur_k.detach()
                cur_v = cur_v.detach()
            k_cache_list.append(cur_k)
            v_cache_list.append(cur_v)

        x = self.norm(x)
        return self.output_proj(x)


class MoDAARPredictorExact(nn.Module):
    """Historical strong-path predictor: LeWM skeleton + MoDA kernel."""

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
        use_moda=True,
        depth_start_layer=1,
        cache_window=0,
        detach_cache=False,
        chunk_visible=False,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.use_moda = use_moda
        self.transformer = MoDATransformerExact(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_moda=use_moda,
            depth_start_layer=depth_start_layer,
            cache_window=cache_window,
            detach_cache=detach_cache,
            chunk_visible=chunk_visible,
        )

    def forward(self, x, c):
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)
