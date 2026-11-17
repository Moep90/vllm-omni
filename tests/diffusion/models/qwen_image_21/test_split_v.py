# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_route_split_v_shard():
    split = {"b.attn.to_qkv.weight": None, "b.attn.to_v.weight": None}
    assert transformer._route_split_v_shard("b.attn.to_qkv.weight", "v", split) == ("b.attn.to_v.weight", None)
    assert transformer._route_split_v_shard("b.attn.to_qkv.weight_scale", "k", split) == (
        "b.attn.to_qkv.weight_scale",
        1,
    )
    assert transformer._route_split_v_shard("b.attn.to_qkv.input_scale", "q", split) == ("b.attn.to_qkv.input_scale", 0)

    fused = {"b.attn.to_qkv.weight": None}
    assert transformer._route_split_v_shard("b.attn.to_qkv.weight", "v", fused) == ("b.attn.to_qkv.weight", "v")
    assert transformer._route_split_v_shard("b.img_mlp.out.weight", None, split) == ("b.img_mlp.out.weight", None)


@pytest.mark.parametrize(
    "excluded,split", [({"blk.attn.to_v"}, True), (set(), False), ({"blk.attn.to_q", "blk.attn.to_v"}, False)]
)
def test_attention_splits_v_when_only_v_is_excluded(monkeypatch, excluded, split):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    quant_config = SimpleNamespace(
        is_layer_excluded=lambda prefix: prefix in excluded,
        get_quant_method=lambda layer, prefix: UnquantizedLinearMethod(),
        online_quantization_config=None,
    )
    attention = transformer.QwenImage21Attention(
        dim=256, heads=2, dim_head=128, quant_config=quant_config, prefix="blk.attn"
    )

    assert attention.split_v is split
    assert attention.to_qkv.weight.shape == (512 if split else 768, 256)
    assert hasattr(attention, "to_v") is split
    assert (attention.num_heads, attention.num_kv_heads) == (2, 2)
