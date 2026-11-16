#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a ComfyUI NVFP4 Qwen-Image-2.1 checkpoint to a ModelOpt W4A16 NVFP4 diffusers folder.

ComfyUI (comfy-kitchen ``TensorCoreNVFP4Layout``) and ModelOpt store the same
NVFP4 numbers differently:

  - FP4 nibbles: ComfyUI puts the even element in the high nibble, ModelOpt
    and vLLM in the low nibble.
  - Block scales (FP8 E4M3, one per 16 inputs): ComfyUI stores them in the
    cuBLAS 128x4 tiled layout, ModelOpt row-major ``[N, K/16]``.
  - The FP32 per-tensor ``weight_scale_2`` means the same in both.
  - ComfyUI files are weight-only (no ``input_scale``), hence W4A16.

Layout differences against vLLM-Omni's Qwen-Image-2.1 modules:

  - ComfyUI fuses ``img_mlp.gate_layer`` and ``img_mlp.proj`` into
    ``img_mlp.gate_up`` ([gate; up] rows). It is split back.
  - vLLM-Omni fuses ``to_q/to_k/to_v`` into one ``to_qkv`` linear with one
    global scale. Blocks where all three are NVFP4 get a shared global scale
    (block scales re-rounded to FP8). Blocks where any shard is BF16 get the
    NVFP4 shards dequantized to BF16 and the whole ``to_qkv`` excluded.
  - The Qwen3-VL text encoder keys move from ``model.layers`` to
    ``model.language_model.layers`` (transformers v5 naming).

Scheduler, processor, VAE and configs come from the base diffusers
checkpoint (``Qwen/Qwen-Image-2.1-Turbo``), which this tool copies.

Usage:
  python vllm_omni/quantization/tools/convert_comfyui_nvfp4_qwen_image_21.py \\
      --base /path/to/Qwen-Image-2.1-Turbo-configs-and-vae \\
      --transformer qwen_image_2.1_turbo_nvfp4.safetensors \\
      --text-encoder qwen3vl_8b_turbo_nvfp4.safetensors \\
      --output /path/to/Qwen-Image-2.1-Turbo-W4A16-NVFP4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
QKV_SHARDS = ("to_q", "to_k", "to_v")
# Base-checkpoint entries copied as they are; the two converted components are written.
BASE_ENTRIES = ("model_index.json", "scheduler", "processor", "vae")


# Headroom over the calibrated amax; a larger global scale only coarsens tiny FP8 block scales.
INPUT_AMAX_MARGIN = 2.0


def quant_config(exclude_modules: list[str], quant_algo: str = "W4A16_NVFP4") -> dict:
    return {
        "quant_method": "modelopt",
        "quant_algo": quant_algo,
        "group_size": 16,
        "exclude_modules": sorted(exclude_modules),
    }


def swap_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return (packed << 4) | (packed >> 4)


def unswizzle_block_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Invert comfy-kitchen ``to_blocked`` (cuBLAS 128x4 tiles) to row-major ``[rows, cols]``."""
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    tiles = blocked.reshape(row_tiles, col_tiles, 32, 4, 4).transpose(2, 3)
    linear = tiles.reshape(row_tiles, col_tiles, 128, 4).permute(0, 2, 1, 3)
    return linear.reshape(row_tiles * 128, col_tiles * 4)[:rows, :cols].contiguous()


def comfy_to_modelopt(weight: torch.Tensor, block_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, packed_cols = weight.shape
    if packed_cols * 2 % 16:
        raise ValueError(f"NVFP4 input width {packed_cols * 2} is not a multiple of 16")
    scale = unswizzle_block_scales(block_scale.view(torch.float8_e4m3fn), rows, packed_cols * 2 // 16)
    return swap_nibbles(weight), scale


def dequantize(weight: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a ModelOpt-layout NVFP4 weight (low nibble = even element) to BF16."""
    lut = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES))
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).flatten(-2).long()
    values = lut[codes].reshape(weight.shape[0], -1, 16)
    scale = block_scale.float() * global_scale.float()
    return (values * scale.unsqueeze(-1)).flatten(1).to(torch.bfloat16)


def share_global_scale(shards: list[dict[str, torch.Tensor]]) -> float:
    """Rewrite each shard's block scales against the largest global scale.

    vLLM keeps one global scale per fused linear (the max over shards), so a
    shard quantized with a smaller one needs its block scales multiplied by
    the ratio. That re-rounds them to FP8, which is the only loss.
    Returns the largest relative error of a re-rounded block scale.
    """
    target = max(float(s["weight_scale_2"]) for s in shards)
    worst = 0.0
    for shard in shards:
        exact = shard["weight_scale"].float() * (float(shard["weight_scale_2"]) / target)
        rounded = exact.clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
        nonzero = exact > 0
        if nonzero.any():
            err = ((rounded.float() - exact).abs() / exact)[nonzero].max().item()
            worst = max(worst, err)
        shard["weight_scale"] = rounded
        shard["weight_scale_2"] = torch.tensor(target, dtype=torch.float32)
    return worst


def _load_linears(sd: dict[str, torch.Tensor]) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Split a ComfyUI state dict into NVFP4 linears (ModelOpt layout) and plain tensors."""
    quantized = {k[: -len(".comfy_quant")] for k in sd if k.endswith(".comfy_quant")}
    linears: dict[str, dict[str, torch.Tensor]] = {}
    for module in quantized:
        fmt = json.loads(bytes(sd[f"{module}.comfy_quant"].tolist()))["format"]
        if fmt != "nvfp4":
            raise ValueError(f"{module}: unsupported ComfyUI quant format {fmt!r}")
        weight, scale = comfy_to_modelopt(sd[f"{module}.weight"], sd[f"{module}.weight_scale"])
        linears[module] = {
            "weight": weight,
            "weight_scale": scale,
            "weight_scale_2": sd[f"{module}.weight_scale_2"].float().reshape(()),
        }
    suffixes = (".comfy_quant", ".weight", ".weight_scale", ".weight_scale_2")
    plain = {k: v for k, v in sd.items() if not any(k == m + s for m in quantized for s in suffixes)}
    return linears, plain


def _emit(out: dict[str, torch.Tensor], module: str, linear: dict[str, torch.Tensor]) -> None:
    for name, tensor in linear.items():
        out[f"{module}.{name}"] = tensor


def _bf16_linears(tensors: dict[str, torch.Tensor]) -> list[str]:
    return [
        k[: -len(".weight")]
        for k, v in tensors.items()
        if k.endswith(".weight") and v.dtype == torch.bfloat16 and v.ndim == 2
    ]


def convert_transformer(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str], float]:
    linears, out = _load_linears(sd)
    worst_scale_err = 0.0

    for module in [m for m in linears if m.endswith(".img_mlp.gate_up")]:
        fused = linears.pop(module)
        half = fused["weight"].shape[0] // 2
        stem = module[: -len("gate_up")]
        for idx, name in enumerate(("gate_layer", "proj")):
            rows = slice(idx * half, (idx + 1) * half)
            _emit(
                out,
                stem + name,
                {
                    "weight": fused["weight"][rows].contiguous(),
                    "weight_scale": fused["weight_scale"][rows].contiguous(),
                    "weight_scale_2": fused["weight_scale_2"].clone(),
                },
            )

    blocks = sorted({m.rsplit(".attn.", 1)[0] for m in linears if m.endswith(QKV_SHARDS)})
    for block in blocks:
        names = [f"{block}.attn.{s}" for s in QKV_SHARDS]
        if all(n in linears for n in names):
            shards = [linears.pop(n) for n in names]
            worst_scale_err = max(worst_scale_err, share_global_scale(shards))
            for name, shard in zip(names, shards):
                _emit(out, name, shard)
        else:
            for name in names:
                if name in linears:
                    q = linears.pop(name)
                    out[f"{name}.weight"] = dequantize(q["weight"], q["weight_scale"], q["weight_scale_2"])

    for module, linear in linears.items():
        _emit(out, module, linear)
    return out, _bf16_linears(out), worst_scale_err


def add_input_scales(tensors: dict[str, torch.Tensor], input_amax: dict[str, float]) -> None:
    """Add ModelOpt ``input_scale`` (amax / (448 * 6)) to every NVFP4 transformer linear, turning it W4A4.

    ``input_amax`` is keyed by runtime prefix (``transformer.<module>``, fused ``to_qkv``).
    """
    for key in [k for k in tensors if k.endswith(".weight_scale_2")]:
        module = key[: -len(".weight_scale_2")]
        runtime = "transformer." + re.sub(r"\.attn\.to_[qkv]$", ".attn.to_qkv", module)
        if runtime not in input_amax:
            raise ValueError(f"no calibrated input amax for {runtime}")
        tensors[f"{module}.input_scale"] = torch.tensor(input_amax[runtime] * INPUT_AMAX_MARGIN / (FP8_MAX * 6.0))


_TE_RENAMES = (
    ("model.layers.", "model.language_model.layers."),
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.norm.", "model.language_model.norm."),
)


def convert_text_encoder(sd: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    linears, plain = _load_linears(sd)
    out = dict(plain)
    for module, linear in linears.items():
        _emit(out, module, linear)

    def rename(key: str) -> str:
        for old, new in _TE_RENAMES:
            if key.startswith(old):
                return new + key[len(old) :]
        return key

    out = {rename(k): v for k, v in out.items()}
    # Vision tower and LM head stay BF16; listing whole subtrees keeps the list short.
    exclude = [m for m in _bf16_linears(out) if not m.startswith("model.visual.")] + ["model.visual*"]
    return out, exclude


def _read(path: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def _write_component(
    out_dir: pathlib.Path,
    base_dir: pathlib.Path,
    name: str,
    weights_file: str,
    tensors: dict[str, torch.Tensor],
    qc: dict,
    qc_file: str | None,
) -> None:
    comp = out_dir / name
    comp.mkdir(parents=True, exist_ok=True)
    config = json.loads((base_dir / name / "config.json").read_text())
    if qc_file is None:
        config["quantization_config"] = qc
    else:
        # Kept beside config.json so transformers does not try to apply it when building the module.
        (comp / qc_file).write_text(json.dumps(qc, indent=2) + "\n")
    (comp / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors, str(comp / weights_file), metadata={"format": "pt"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base", required=True, help="Diffusers Qwen-Image-2.1 folder with configs, processor, scheduler, vae"
    )
    parser.add_argument("--transformer", required=True, help="ComfyUI NVFP4 diffusion model .safetensors")
    parser.add_argument("--text-encoder", required=True, help="ComfyUI NVFP4 Qwen3-VL text encoder .safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--input-amax",
        help="JSON from calibrate_nvfp4_input_amax.py; makes the transformer W4A4 (NVFP4) instead of W4A16",
    )
    args = parser.parse_args()

    base, out = pathlib.Path(args.base), pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for entry in BASE_ENTRIES:
        src = base / entry
        if src.is_dir():
            shutil.copytree(src, out / entry, dirs_exist_ok=True)
        else:
            shutil.copy2(src, out / entry)

    tensors, exclude, worst = convert_transformer(_read(args.transformer))
    transformer_algo = "W4A16_NVFP4"
    if args.input_amax:
        add_input_scales(tensors, json.loads(pathlib.Path(args.input_amax).read_text()))
        transformer_algo = "NVFP4"
    _write_component(
        out,
        base,
        "transformer",
        "diffusion_pytorch_model.safetensors",
        tensors,
        quant_config(exclude, transformer_algo),
        None,
    )
    print(
        f"transformer: {len(tensors)} tensors, {len(exclude)} BF16 linears, "
        f"max shared-global-scale re-rounding error {worst:.4f}"
    )
    del tensors

    tensors, exclude = convert_text_encoder(_read(args.text_encoder))
    _write_component(
        out, base, "text_encoder", "model.safetensors", tensors, quant_config(exclude), "hf_quant_config.json"
    )
    print(f"text_encoder: {len(tensors)} tensors, {len(exclude)} excluded entries")


if __name__ == "__main__":
    main()
