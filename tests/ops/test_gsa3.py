# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.utils import assert_close, device, is_intel_alchemist
from fla.ops.gsa3 import chunk_gsa3



def recurrent_gsa3_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale_k: float = None,
    scale_v: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, w, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, w, beta, g])
    B, H, T, K, V, M = *q.shape, v.shape[-1], w.shape[-1]
    ok = torch.zeros(B, H, T, M).to(v)
    hk = torch.zeros(B, H, K, M).to(v)
    if initial_state is not None:
        hk += initial_state[0]
    if scale_k is None:
        scale_k = 1 / (q.shape[-1] ** 0.5)
    if scale_v is None:
        scale_v = 1 / (w.shape[-1] ** 0.5)
    q = q * scale_k
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = w[:, :, i] # B H D
        g_i = g[:, :, i]
        hk = hk * g_i.exp()[:, :, :, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (hk * b_k[:, :, :, None]).sum(-2)
        b_k = b_k * b_beta[..., None]
        hk = hk + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        ok[:, :, i] = (b_q[..., None] * hk).sum(-2)
        
    qv = ok.softmax(-1)
    qv = qv * scale_v
    ov = torch.zeros(B, H, T, V).to(v)
    hv = torch.zeros(B, H, M, V).to(v)
    if initial_state is not None:
        hv += initial_state[1]
    for i in range(T):
        b_q = qv[:, :, i]
        b_k = w[:, :, i]
        b_v = v[:, :, i] # B H D
        g_i = g[:, :, i]
        hv = hv * g_i.exp()[:, :, None, :]
        b_beta = beta[:, :, i]
        b_k = b_k - (hv * b_v[:, :, None, :]).sum(-1)
        b_v = b_v * b_beta[..., None]
        hv = hv + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        ov[:, :, i] = (b_q[..., None] * hv).sum(-2)    
        
    if output_final_state:
        h = (hk, hv)
    else:
        h = None
    ov = ov.transpose(1, 2).contiguous()
    return ov, h


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale_k', 'scale_v', 'gate_logit_normalizer', 'dtype'),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-scalek{}-scalev{}-gate_logit_normalizer{}-{}".format(*test))
        for test in [
            (4, 1024, 8, 128, 1, 1, 1, torch.float16),
            (4, 2048, 16, 128, 0.1, 0.1, 10, torch.float16),
            (4, 3407, 16, 128, 0.1, 1, 1, torch.float16),
            (4, 3407, 16, 128, 0.1, 1, 0.1, torch.float16),
            (4, 2048, 4, 128, 1, 0.1, 1, torch.float16),
            (4, 1560, 8, 192, 0.1, 0.1, 1, torch.float16),
            (2, 1024, 4, 128, 0.1, 0.1, 1, torch.float16),
            (2, 2048, 4, 128, 0.1, 0.1, 0.1, torch.float16),
            (3, 1500, 4, 256, 0.1, 0.1, 10, torch.float16)
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale_k: float,
    scale_v: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    w = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=torch.float32).sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, D, dtype=torch.float32)) / gate_logit_normalizer
    hk0 = torch.randn(B, H, D, D, dtype=torch.float32)
    hv0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, w, beta, g, hk0, hv0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, w, beta, g, hk0, hv0))
    do = torch.randn_like(v)
    dhkt = torch.randn_like(hk0)
    dhvt = torch.randn_like(hv0)
    ref, (ref_hkt, ref_hvt) = recurrent_gsa3_ref(
        q=q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=F.normalize(v.clone(), p=2, dim=-1),
        w=w.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale_k=scale_k,
        scale_v=scale_v,
        initial_state=(hk0.clone(), hv0.clone()),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_hkt * dhkt).sum() + (ref_hvt * dhvt).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dw, ref_dbeta, ref_dg, ref_dhk0, ref_dhv0 = q.grad, k.grad, v.grad, w.grad, beta.grad, g.grad, hk0.grad, hv0.grad
    q.grad = k.grad = v.grad = w.grad = beta.grad = g.grad = hk0.grad = hv0.grad = None
    
    tri, (tri_hkt, tri_hvt) = chunk_gsa3(
        q=q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=F.normalize(v.clone(), p=2, dim=-1),
        w=w.clone(),
        beta=beta.clone(),
        g=g.clone(),
        scale_k=scale_k,
        scale_v=scale_v,
        initial_state=(hk0.clone(), hv0.clone()),
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_hkt * dhkt).sum() + (tri_hvt * dhvt).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dw, tri_dbeta, tri_dg, tri_dhk0, tri_dhv0 = q.grad, k.grad, v.grad, w.grad, beta.grad, g.grad, hk0.grad, hv0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('hk', ref_hkt, tri_hkt, 0.005)
    assert_close('hv', ref_hvt, tri_hvt, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.005)
    assert_close('dk', ref_dk, tri_dk, 0.005)
    assert_close('dv', ref_dv, tri_dv, 0.005)
    assert_close('dw', ref_dw, tri_dw, 0.005)
    assert_close('dbeta', ref_dbeta, tri_dbeta, 0.005)
    assert_close('dhk0', ref_dhk0, tri_dhk0, 0.005)
    assert_close('dhv0', ref_dhv0, tri_dhv0, 0.005)
    assert_close('dg', ref_dg, tri_dg, 0.005)
    