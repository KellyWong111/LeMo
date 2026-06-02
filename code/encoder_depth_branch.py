from types import SimpleNamespace

import torch
import torch.nn as nn
from einops import rearrange


class LayerDepthAttention(nn.Module):
    """Attend over encoder layers for each token position independently."""

    def __init__(self, dim, heads=3, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, q_tokens, layer_tokens):
        """
        q_tokens:    [B, N, D]     query from final encoder layer
        layer_tokens:[B, L, N, D]  per-layer encoder tokens
        """
        b, l, n, d = layer_tokens.shape

        q = self.to_q(self.norm_q(q_tokens))
        q = rearrange(q, "b n (h d) -> (b n) h 1 d", h=self.heads)

        kv = self.norm_kv(layer_tokens)
        k = self.to_k(kv)
        v = self.to_v(kv)
        k = rearrange(k, "b l n (h d) -> (b n) h l d", h=self.heads)
        v = rearrange(v, "b l n (h d) -> (b n) h l d", h=self.heads)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        out = rearrange(out, "(b n) h 1 d -> b n (h d)", b=b, n=n)
        return self.to_out(out)


class EncoderDepthBranchViT(nn.Module):
    """
    Wrap a ViT encoder and add a non-causal cross-layer depth branch on top.

    The base ViT still performs all within-layer token self-attention.
    The added branch only changes how layerwise visual representations are fused.
    """

    def __init__(
        self,
        vit,
        *,
        hidden_size,
        depth_branch_heads=3,
        depth_branch_dim_head=64,
        depth_branch_dropout=0.0,
        use_depth_branch=True,
        layer_start=0,
        layer_end=None,
        include_final_layer=False,
        init_gate=0.0,
    ):
        super().__init__()
        self.vit = vit
        self.config = vit.config
        self.vit.config.output_hidden_states = True
        self.use_depth_branch = bool(use_depth_branch)
        self.layer_start = int(layer_start)
        self.layer_end = layer_end
        self.include_final_layer = bool(include_final_layer)
        self.depth_attn = LayerDepthAttention(
            hidden_size,
            heads=depth_branch_heads,
            dim_head=depth_branch_dim_head,
            dropout=depth_branch_dropout,
        )
        self.depth_gate = nn.Parameter(torch.tensor(float(init_gate)))

    def _select_hidden_states(self, hidden_states):
        # Drop patch-embedding state and, by default, exclude the final block output
        # so the depth branch only retrieves from earlier encoder layers.
        layer_states = list(hidden_states[1:])
        if not self.include_final_layer:
            layer_states = layer_states[:-1]
        end = self.layer_end if self.layer_end is not None else len(layer_states)
        layer_states = layer_states[self.layer_start:end]
        if not layer_states:
            raise ValueError(
                "EncoderDepthBranchViT selected no hidden states. "
                f"Got layer_start={self.layer_start}, layer_end={self.layer_end}, "
                f"include_final_layer={self.include_final_layer}."
            )
        return torch.stack(layer_states, dim=1)

    def _forward_vit_with_layer_states(self, pixels, interpolate_pos_encoding=True):
        embeddings = self.vit.embeddings(
            pixels,
            bool_masked_pos=None,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        layer_states = [embeddings]
        hidden_states = embeddings
        for layer_module in self.vit.encoder.layer:
            hidden_states = layer_module(hidden_states)
            layer_states.append(hidden_states)

        last_hidden = self.vit.layernorm(hidden_states)
        return last_hidden, tuple(layer_states)

    def forward(self, pixels, interpolate_pos_encoding=True):
        last_hidden, hidden_states = self._forward_vit_with_layer_states(
            pixels, interpolate_pos_encoding=interpolate_pos_encoding
        )
        pooled = last_hidden[:, 0]

        if not self.use_depth_branch:
            return SimpleNamespace(last_hidden_state=last_hidden, pooled_state=pooled)

        layer_tokens = self._select_hidden_states(hidden_states)
        depth_out = self.depth_attn(last_hidden, layer_tokens)
        fused_hidden = last_hidden + self.depth_gate * depth_out
        depth_out_norm = depth_out.norm(dim=-1).mean()
        fused_delta_norm = (fused_hidden - last_hidden).norm(dim=-1).mean()
        return SimpleNamespace(
            last_hidden_state=fused_hidden,
            pooled_state=fused_hidden[:, 0],
            depth_gate=self.depth_gate.detach(),
            depth_out_norm=depth_out_norm.detach(),
            fused_delta_norm=fused_delta_norm.detach(),
            depth_num_layers=torch.tensor(
                layer_tokens.size(1), device=fused_hidden.device, dtype=fused_hidden.dtype
            ),
        )
