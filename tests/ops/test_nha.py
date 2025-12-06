import os
from typing import List, Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.utils import assert_close, device, is_intel_alchemist

from fla.ops.gsa import chunk_gsa
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False



def naive_recurrent_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> torch.Tensor:
    dtype = q.dtype
    q, k, v, s, g = map(lambda x: x.transpose(1, 2).contiguous().float(), (q, k, v, s, g))

    NG = q.shape[1]//k.shape[1]
    # [batch_size, n_heads, seq_len, n_slots]
    if g is None:
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s = torch.exp(s - z)
    k, v, s, g = map(lambda x: repeat(x, 'b h t d -> b (h g) t d', g=NG), (k, v, s, g))
    if initial_state is not None:
        initial_state = tuple(map(lambda x: repeat(x, 'b h k v -> b (h g) k v', g=NG), initial_state))

    B, H, T, K, V, M = *q.shape, v.shape[-1], s.shape[-1]

    hk = torch.zeros(B, H, K, M, dtype=torch.float, device=q.device)
    ok = torch.zeros_like(s)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    final_state = None
    if initial_state is not None:
        hk += initial_state[0]

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = s[:, :, i]
        g_i = g[:, :, i].exp()
        hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
        ok[:, :, i] = (q_i[..., None] * hk).sum(-2)

    qv = ok.softmax(-1)
    hv = torch.zeros(B, H, M, V, dtype=torch.float, device=q.device)
    ov = torch.zeros_like(v)
    if initial_state is not None:
        hv += initial_state[1]

    for i in range(T):
        q_i = qv[:, :, i]
        k_i = s[:, :, i]
        v_i = v[:, :, i]
        g_i = g[:, :, i].exp()
        hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
        ov[:, :, i] = (q_i[..., None] * hv).sum(-2)

    if output_final_state:
        final_state = (hk.view(B, -1, NG, K, M)[:, :, 0], hv.view(B, -1, NG, M, V)[:, :, 0])
    ov = ov.transpose(1, 2).contiguous()
    return ov.to(dtype), final_state



@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'gate_logit_normalizer', 'mask_p', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-{}".format(*test)
        )
        for test in [
            (4, 4096, 4, 128, 1, 1, 0, torch.float16),
            (1, 4096, 1, 64, 1, 1, 0, torch.float16),
            (2, 4096, 3, 60, 1, 1, 0, torch.float16),
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    q = torch.rand(B, T, H, D, dtype=dtype)
    k = torch.rand(B, T, H, D, dtype=dtype)
    v = torch.rand(B, T, H, D, dtype=torch.float)
    beta = torch.rand(B, T, H, dtype=torch.float).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float)) / gate_logit_normalizer
    g = g * (torch.rand_like(g) > mask_p)
    s = torch.rand(B, T, H, D, dtype=torch.float)
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, s, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, s, g, h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    print('================== Running forward and backward ==================')

    o1, lse1, _ = flash_attn_func(
        q=q,
        k=k,
        v=v,
        return_attn_probs=True
    )
    
    tri, tri_ht = naive_recurrent_gsa(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        s=s.clone(),
        g=g.clone(),
    )
    
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = g.grad = beta.grad = h0.grad = None
    
    # breakpoint()

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    


@pytest.mark.parametrize(
    ('H', 'D', 'mask_p', 'cu_seqlens', 'dtype'),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (4, 60, 0, [0, 96, 177], torch.float16),
            (16, 128, 0, [0, 256, 500, 1000], torch.float16),
            (4, 64, 0.5, [0, 256, 500, 1000], torch.float16),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ]
)
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '1',
    reason='Skipping test_chunk_varlen because SKIP_TEST_CHUNK_VARLEN is set'
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    if is_intel_alchemist and D > 128:
        pytest.skip(reason='chunk_gated_delta_rule is not supported on alchemist for D>128')
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, H, D), dtype=dtype)
    k = torch.randn((1, T, H, D), dtype=dtype)
    v = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    g = F.logsigmoid(torch.rand(1, T, H, D, dtype=torch.float))
    g = g * (torch.rand_like(g) > mask_p)
    beta = torch.rand(1, T, H, dtype=torch.float32).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=torch.float32)

    q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0))
    do = torch.randn_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_oja2(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        gv=g.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = recurrent_oja2_ref(
            q=q[:, cu_seqlens[i]:cu_seqlens[i+1]],
            k=k[:, cu_seqlens[i]:cu_seqlens[i+1]],
            v=v[:, cu_seqlens[i]:cu_seqlens[i+1]],
            beta=beta[:, cu_seqlens[i]:cu_seqlens[i+1]],
            g=g[:, cu_seqlens[i]:cu_seqlens[i+1]],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)
    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dg', ref_dg, tri_dg, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
