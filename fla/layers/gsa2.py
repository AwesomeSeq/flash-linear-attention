# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gsa2 import chunk_gsa2
from fla.ops.kda.gate import fused_kda_gate
from fla.modules.l2norm import l2_norm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedSlotAttention2(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        num_slots: Optional[int] = None,
        scale_k: Optional[int] = None,
        scale_v: Optional[int] = None,
        use_w_lora: bool = False,
        use_kda_gate: bool = False,
        use_qk_norm: bool = False,
        **kwargs,
    ) -> GatedSlotAttention2:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots
        
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.w_dim = int(self.num_heads * num_slots)
        self.layer_idx = layer_idx
        self.use_kda_gate = use_kda_gate
        self.scale_k = scale_k
        self.scale_v = scale_v
        self.use_qk_norm = use_qk_norm

        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.w_conv1d = ShortConvolution(
                hidden_size=self.w_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        if use_w_lora:
            self.w_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.w_dim, bias=False),
            )
        else:
            self.w_proj = nn.Linear(hidden_size, self.w_dim, bias=False)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.w_dim, bias=False),
        )
        if use_kda_gate:
            self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
            self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v, conv_state_w = None, None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v, conv_state_w = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            w, conv_state_w = self.v_conv1d(
                x=self.w_proj(hidden_states),
                cache=conv_state_w,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        if self.use_kda_gate:
            g = fused_kda_gate(g, self.A_log, self.head_k_dim, g_bias=self.dt_bias)
        else:
            g = F.logsigmoid(g)
            g = rearrange(g, '... (h d) -> ... h d', d=self.num_slots)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        w = rearrange(w, '... (h d) -> ... h d', d=self.num_slots)
        w = l2_norm(w)
        if self.use_qk_norm:
            q = l2_norm(q)
            k = l2_norm(k)

        if self.num_v_heads > self.num_heads:
            q, k = (repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads) for x in (q, k))

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gsa2(
                q=q,
                k=k,
                v=v,
                w=w,
                g=g,
                beta=beta,
                scale_k=self.scale_k,
                scale_v=self.scale_v,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        elif mode == 'fused_recurrent':
            assert NotImplementedError(f"ohhh, gsa2 only support training mode for now.")
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v, conv_state_w) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim))
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
