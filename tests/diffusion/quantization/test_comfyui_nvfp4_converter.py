# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.tools import convert_comfyui_nvfp4_qwen_image_21 as conv

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _to_blocked(m: torch.Tensor) -> torch.Tensor:
    # Verbatim comfy-kitchen 0.2.40 float_utils.to_blocked(flatten=False), the layout ComfyUI writes.
    rows, cols = m.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros((rb * 128, cb * 4), dtype=m.dtype)
    padded[:rows, :cols] = m
    blocks = padded.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).reshape(rb * 128, cb * 4)


def _comfy_linear(rows: int, cols: int, global_scale: float, seed: int):
    """Random NVFP4 linear in ComfyUI layout plus its exact float value."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.uint8)
    scales = (torch.rand(rows, cols // 16, generator=g) * 400 + 1).to(torch.float8_e4m3fn)
    packed = (codes[:, 0::2] << 4) | codes[:, 1::2]  # hi_first=True, comfy-kitchen's default
    value = E2M1[codes.long()] * scales.float().repeat_interleave(16, dim=1) * global_scale
    tensors = {
        "weight": packed,
        "weight_scale": _to_blocked(scales),
        "weight_scale_2": torch.tensor(global_scale, dtype=torch.float32),
        "comfy_quant": torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8),
    }
    return tensors, value


def _add(sd, module, tensors):
    for name, t in tensors.items():
        sd[f"{module}.{name}"] = t


@pytest.mark.parametrize("rows,cols", [(256, 8), (200, 6)])
def test_unswizzle_inverts_comfy_blocked_layout(rows, cols):
    scales = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)
    assert torch.equal(conv.unswizzle_block_scales(_to_blocked(scales), rows, cols), scales)


def test_converted_layout_matches_vllm_reference_dequant():
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import dequantize_to_dtype

    comfy, value = _comfy_linear(256, 128, 0.0123, seed=0)
    weight, scale = conv.comfy_to_modelopt(comfy["weight"], comfy["weight_scale"])
    assert scale.shape == (256, 8) and scale.dtype == torch.float8_e4m3fn

    ours = conv.dequantize(weight, scale, comfy["weight_scale_2"])
    vllm = dequantize_to_dtype(weight, scale, comfy["weight_scale_2"], torch.float32, swizzle=False)
    assert torch.equal(ours, value.to(torch.bfloat16))
    assert torch.allclose(vllm, value, rtol=1e-6, atol=0)


def _transformer_sd():
    sd, values = {}, {}
    for block, v_quantized in ((0, True), (1, False)):
        p = f"transformer_blocks.{block}"
        for i, shard in enumerate(("to_q", "to_k", "to_v")):
            if shard == "to_v" and not v_quantized:
                sd[f"{p}.attn.to_v.weight"] = torch.randn(64, 64).to(torch.bfloat16)
                continue
            t, values[f"{p}.attn.{shard}"] = _comfy_linear(64, 64, 0.01 * (i + 1), seed=10 * block + i)
            _add(sd, f"{p}.attn.{shard}", t)
        t, values[f"{p}.img_mlp.gate_up"] = _comfy_linear(256, 64, 0.02, seed=100 + block)
        _add(sd, f"{p}.img_mlp.gate_up", t)
        sd[f"{p}.attn.to_out.0.weight"] = torch.randn(64, 64).to(torch.bfloat16)
        sd[f"{p}.attn.norm_q.weight"] = torch.ones(16, dtype=torch.bfloat16)
    sd["proj_out.weight"] = torch.randn(4, 64).to(torch.bfloat16)
    return sd, values


def test_convert_transformer_layout():
    sd, values = _transformer_sd()
    out, exclude, worst = conv.convert_transformer(sd)

    assert not any(k.endswith(".comfy_quant") or ".gate_up." in k for k in out)
    # gate_up rows are [gate; up]: the first half is gate_layer, the second proj.
    gate_up = values["transformer_blocks.0.img_mlp.gate_up"]
    for name, rows in (("gate_layer", slice(0, 128)), ("proj", slice(128, 256))):
        m = f"transformer_blocks.0.img_mlp.{name}"
        got = conv.dequantize(out[f"{m}.weight"], out[f"{m}.weight_scale"], out[f"{m}.weight_scale_2"])
        assert torch.equal(got, gate_up[rows].to(torch.bfloat16))

    # Block 0: all shards NVFP4, rewritten to one global scale with bounded FP8 re-rounding.
    for s in ("to_q", "to_k", "to_v"):
        assert out[f"transformer_blocks.0.attn.{s}.weight_scale_2"].item() == pytest.approx(0.03)
    assert 0 < worst <= 2**-4
    for s in ("to_q", "to_k", "to_v"):
        m = f"transformer_blocks.0.attn.{s}"
        got = conv.dequantize(out[f"{m}.weight"], out[f"{m}.weight_scale"], out[f"{m}.weight_scale_2"]).float()
        ref = values[m]
        assert ((got - ref).abs() <= ref.abs() * 2**-4 + 1e-6).all()

    # Block 1: to_v is BF16, so to_q/to_k are dequantized exactly and the fused to_qkv stays BF16.
    for s in ("to_q", "to_k"):
        m = f"transformer_blocks.1.attn.{s}"
        assert f"{m}.weight_scale" not in out
        assert torch.equal(out[f"{m}.weight"], values[m].to(torch.bfloat16))
    assert set(exclude) == {
        "transformer_blocks.1.attn.to_q",
        "transformer_blocks.1.attn.to_k",
        "transformer_blocks.1.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.1.attn.to_out.0",
        "proj_out",
    }


def test_convert_text_encoder_renames_and_excludes():
    sd = {}
    t, _ = _comfy_linear(64, 64, 0.01, seed=7)
    _add(sd, "model.layers.0.self_attn.q_proj", t)
    sd["model.layers.1.mlp.up_proj.weight"] = torch.randn(64, 64).to(torch.bfloat16)
    sd["model.embed_tokens.weight"] = torch.randn(8, 64).to(torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
    sd["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(48, 16).to(torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(8, 64).to(torch.bfloat16)

    out, exclude = conv.convert_text_encoder(sd)

    assert set(out) == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale_2",
        "model.language_model.layers.1.mlp.up_proj.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "lm_head.weight",
    }
    assert set(exclude) == {
        "model.language_model.layers.1.mlp.up_proj",
        "model.language_model.embed_tokens",
        "lm_head",
        "model.visual*",
    }


def test_quant_config_selects_w4a16_and_excludes_runtime_prefixes():
    _, exclude, _ = conv.convert_transformer(_transformer_sd()[0])
    cfg = build_quant_config(json.loads(json.dumps(conv.quant_config(exclude))))
    assert cfg.get_name() == "modelopt_fp4"
    assert cfg.quant_method == "W4A16_NVFP4"

    # Prefixes as QwenImage21Transformer2DModel builds them (default prefix "transformer").
    assert cfg.is_layer_excluded("transformer.transformer_blocks.1.attn.to_qkv")
    assert not cfg.is_layer_excluded("transformer.transformer_blocks.0.attn.to_qkv")
    assert cfg.is_layer_excluded("transformer.transformer_blocks.0.attn.to_out.0")
    assert not cfg.is_layer_excluded("transformer.transformer_blocks.0.img_mlp.proj")
    assert not cfg.is_layer_excluded("transformer.transformer_blocks.11.attn.to_qkv")

    te = build_quant_config(
        conv.quant_config(["model.language_model.layers.1.mlp.up_proj", "lm_head", "model.visual*"])
    )
    assert te.is_layer_excluded("model.visual.blocks.0.attn.qkv")
    assert te.is_layer_excluded("lm_head")
    assert te.is_layer_excluded("model.language_model.layers.1.mlp.up_proj")
    assert not te.is_layer_excluded("model.language_model.layers.11.mlp.up_proj")


def test_add_input_scales_uses_runtime_prefixes():
    out, _, _ = conv.convert_transformer(_transformer_sd()[0])
    amax = {
        "transformer.transformer_blocks.0.attn.to_qkv": 3.0,
        "transformer.transformer_blocks.0.img_mlp.gate_layer": 5.0,
        "transformer.transformer_blocks.0.img_mlp.proj": 5.0,
        "transformer.transformer_blocks.1.img_mlp.gate_layer": 7.0,
        "transformer.transformer_blocks.1.img_mlp.proj": 7.0,
    }
    conv.add_input_scales(out, amax)

    expected = 3.0 * conv.INPUT_AMAX_MARGIN / 2688
    for s in ("to_q", "to_k", "to_v"):
        assert out[f"transformer_blocks.0.attn.{s}.input_scale"].item() == pytest.approx(expected)
    # BF16 (dequantized) shards get no input scale.
    assert "transformer_blocks.1.attn.to_q.input_scale" not in out
    with pytest.raises(ValueError, match="no calibrated input amax"):
        conv.add_input_scales(out, {})
