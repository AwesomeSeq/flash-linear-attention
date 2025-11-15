# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.gsa2.kda.chunk import chunk_kda_bwd, chunk_kda_fwd
from fla.ops.gsa2.oja2.chunk import chunk_oja2_bwd, chunk_oja2_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_local_cumsum
from fla.ops.utils.op import exp
from fla.ops.utils.softmax import softmax_bwd, softmax_fwd
from fla.utils import autotune_cache_kwargs, input_guard


def chunk_gsa3_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    output_final_state: bool = False,
    scale_k: float = 1.,
    scale_v: float = 1.,
    cu_seqlens: Optional[torch.LongTensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    _, o_delta, Aqk, Akk, hkt = chunk_kda_fwd(
        q=q,
        k=k,
        v=w,
        g=g,
        beta=beta,
        scale=scale_k,
        initial_state=hk0,
        output_final_state=output_final_state,
        g_cumsum=False,
        cu_seqlens=cu_seqlens
    )

    # p is kept in fp32 for safe softmax backward
    p = softmax_fwd(o_delta, dtype=torch.float)
    qv = p.to(q.dtype)

    _, o_oja, Avv, hvt = chunk_oja2_fwd(
        q=qv,
        k=w,
        v=v,
        gv=g,
        beta=beta,
        scale=scale_v,
        initial_state=hv0,
        output_final_state=output_final_state,
        g_cumsum=False,
        cu_seqlens=cu_seqlens,
    )

    return o_oja, p, Avv, Aqk, Akk, hkt, hvt

def chunk_gsa3_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    o_oja: torch.Tensor,
    p: torch.Tensor,
    Avv: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    scale_k: float,
    scale_v: float,
    do: torch.Tensor,
    dht: Tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None
):
    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state

    dhkt, dhvt = dht

    qv = p.to(q.dtype)
    dqv, dw_oja, dv, db_oja, dgv, dhv0 = chunk_oja2_bwd(
        q=qv,
        k=w,
        v=v,
        gv=g,
        beta=beta,
        A=Avv,
        o=o_oja,
        scale=scale_v,
        initial_state=hv0,
        do=do,
        dht=dhvt,
        dgk=None,
        cu_seqlens=cu_seqlens
    )

    # softmax gradient, equivalent to:
    # dok = qv * (dqv - (qv * dqv).sum(-1, True))
    do_kda = softmax_bwd(p, dqv, dtype=o_oja.dtype)

    dq, dk, dw, db, dg, dhk0 = chunk_kda_bwd(
        q=q,
        k=k,
        v=w,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale_k,
        initial_state=hk0,
        do=do_kda,
        dht=dhkt,
        cu_seqlens=cu_seqlens
    )

    dg = dg.add_(dgv)
    dw = dw.add_(dw_oja)
    db = db.add_(db_oja)
    return dq, dk, dv, dw, dg, db, dhk0, dhv0


class ChunkGSA3Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale_k: float,
        scale_v: float,
        hk0: Optional[torch.Tensor],
        hv0: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
        o_oja, p, Avv, Aqk, Akk, hkt, hvt = chunk_gsa3_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            g=g,
            beta=beta,
            initial_state=(hk0, hv0),
            output_final_state=output_final_state,
            scale_k=scale_k,
            scale_v=scale_v,
            cu_seqlens=cu_seqlens
        )

        ctx.save_for_backward(q, k, v, w, g, beta, o_oja, p, Avv, Aqk, Akk, hk0, hv0)
        ctx.scale_k = scale_k
        ctx.scale_v = scale_v
        ctx.cu_seqlens = cu_seqlens
        return o_oja, hkt, hvt

    @staticmethod
    @input_guard
    def backward(ctx, dov, dhkt=None, dhvt=None):
        q, k, v, w, g, beta, o_oja, p, Avv, Aqk, Akk, hk0, hv0 = ctx.saved_tensors
        scale_k = ctx.scale_k
        scale_v = ctx.scale_v
        cu_seqlens = ctx.cu_seqlens
        
        dq, dk, dv, dw, dg, dbeta, dhk0, dhv0 = chunk_gsa3_bwd(
            q=q,
            k=k,
            v=v,
            w=w,
            g=g,
            beta=beta,
            o_oja=o_oja,
            p=p,
            Avv=Avv,
            Aqk=Aqk,
            Akk=Akk,
            initial_state=(hk0, hv0),
            scale_k=scale_k,
            scale_v=scale_v,
            do=dov,
            dht=(dhkt, dhvt),
            cu_seqlens=cu_seqlens
        )
        return dq, dk, dv, dw, dg, dbeta, None, None, dhk0, dhv0, None, None


@torch.compiler.disable
def chunk_gsa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale_k: Optional[int] = None,
    scale_v: Optional[int] = None,
    initial_state: Optional[Tuple[torch.Tensor]] = None,
    output_final_state: Optional[bool] = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state[0].shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state[0].shape[0]}."
            )

    hk0, hv0 = None, None
    if initial_state is not None:
        hk0, hv0 = initial_state
    if scale_k is None:
        scale_k = q.shape[-1] ** -0.5
    if scale_v is None:
        scale_v = w.shape[-1] ** -0.5
    o, *final_state = ChunkGSA3Function.apply(
        q,
        k,
        v,
        w,
        g,
        beta,
        scale_k,
        scale_v,
        hk0,
        hv0,
        output_final_state,
        cu_seqlens
    )
    return o, final_state
