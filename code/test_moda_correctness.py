"""Validate that LeWM routes through the official MoDA Triton kernel."""

import math
import sys

import torch

sys.path.insert(0, "/home/internship/wm_transfer_lab/MoDA/libs/moda_triton")
from fla.ops.moda import naive_mixture_of_depth_causal_ref, parallel_moda

from moda_module import MoDAAttention


def _randn(*shape, device, dtype):
    return torch.randn(*shape, device=device, dtype=dtype)


def test_wrapper_matches_direct_kernel_with_depth():
    """LeWM's wrapper should be exactly the same call as official parallel_moda."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, D, L = 2, 3, 4, 64, 3
    scale = 1.0 / math.sqrt(D)

    q = _randn(B, T, H, D, device=device, dtype=dtype)
    k = _randn(B, T, H, D, device=device, dtype=dtype)
    v = _randn(B, T, H, D, device=device, dtype=dtype)
    cached_k = _randn(B, T * L, H, D, device=device, dtype=dtype)
    cached_v = _randn(B, T * L, H, D, device=device, dtype=dtype)

    attn = MoDAAttention(dim=H * D, heads=H, dim_head=D, dropout=0.0).to(device)
    wrapped = attn._parallel_moda(q, k, v, cached_k, cached_v)
    direct = parallel_moda(
        q,
        k,
        v,
        cached_k=cached_k,
        cached_v=cached_v,
        scale=scale,
        moda_group_num=1,
        head_first=False,
        need_lse=False,
        warn_shape=False,
    )

    diff = float((wrapped - direct).abs().max())
    print(f"Wrapper vs direct kernel (depth): max_abs={diff:.6e}")
    assert diff == 0.0


def test_wrapper_matches_direct_kernel_without_depth():
    """Even shallow layers should route through official parallel_moda."""
    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, D = 2, 3, 4, 64
    scale = 1.0 / math.sqrt(D)

    q = _randn(B, T, H, D, device=device, dtype=dtype)
    k = _randn(B, T, H, D, device=device, dtype=dtype)
    v = _randn(B, T, H, D, device=device, dtype=dtype)

    attn = MoDAAttention(dim=H * D, heads=H, dim_head=D, dropout=0.0).to(device)
    wrapped = attn._parallel_moda(q, k, v, cached_k=None, cached_v=None)
    direct = parallel_moda(
        q,
        k,
        v,
        cached_k=None,
        cached_v=None,
        scale=scale,
        moda_group_num=1,
        head_first=False,
        need_lse=False,
        warn_shape=False,
    )

    diff = float((wrapped - direct).abs().max())
    print(f"Wrapper vs direct kernel (no depth): max_abs={diff:.6e}")
    assert diff == 0.0


def test_official_kernel_matches_reference_with_official_tolerance():
    """The official kernel should stay within the tolerance regime used for kernels."""
    torch.manual_seed(7)
    device = "cuda"
    dtype = torch.bfloat16
    B, T, H, D, L = 2, 3, 16, 64, 63
    scale = 1.0 / math.sqrt(D)

    q = _randn(B, T, H, D, device=device, dtype=dtype)
    k = _randn(B, T, H, D, device=device, dtype=dtype)
    v = _randn(B, T, H, D, device=device, dtype=dtype)
    cached_k = _randn(B, T * L, H, D, device=device, dtype=dtype)
    cached_v = _randn(B, T * L, H, D, device=device, dtype=dtype)

    ref_out, _ = naive_mixture_of_depth_causal_ref(
        q, k, v, kd=cached_k, vd=cached_v, scale=scale, moda_group_num=1
    )
    kernel_out = parallel_moda(
        q,
        k,
        v,
        cached_k=cached_k,
        cached_v=cached_v,
        scale=scale,
        moda_group_num=1,
        head_first=False,
        need_lse=False,
        warn_shape=False,
    )

    diff = (kernel_out.float() - ref_out.float()).abs()
    max_abs = float(diff.max())
    l2_rel = float(
        diff.pow(2).sum().sqrt() / (ref_out.float().pow(2).sum().sqrt() + 1e-12)
    )
    print(f"Kernel vs reference: max_abs={max_abs:.6e} l2_rel={l2_rel:.6e}")

    assert max_abs < 1e-2
    assert l2_rel < 1e-2


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Wrapper matches official kernel with depth")
    print("=" * 60)
    test_wrapper_matches_direct_kernel_with_depth()

    print()
    print("=" * 60)
    print("Test 2: Wrapper matches official kernel without depth")
    print("=" * 60)
    test_wrapper_matches_direct_kernel_without_depth()

    print()
    print("=" * 60)
    print("Test 3: Official kernel stays within kernel/reference tolerance")
    print("=" * 60)
    test_official_kernel_matches_reference_with_official_tolerance()

    print()
    print("All correctness tests passed!")
