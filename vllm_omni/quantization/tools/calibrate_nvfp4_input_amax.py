#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record the input amax of every NVFP4 linear of a W4A16 ModelOpt diffusion checkpoint.

Runs the pipeline in-process over a prompt set and keeps, per linear prefix,
the largest |x| any forward saw (all prompts, sizes and denoising steps).
``convert_comfyui_nvfp4_qwen_image_21.py --input-amax`` turns the result into
static ``input_scale`` tensors, which lets vLLM run the layers as W4A4 on FP4
tensor cores (ModelOpt max calibration; ComfyUI computes the same amax per call).

Usage:
  python vllm_omni/quantization/tools/calibrate_nvfp4_input_amax.py \\
      --model /path/to/Qwen-Image-2.1-Turbo-W4A16-NVFP4 --output input_amax.json
"""

from __future__ import annotations

import argparse
import json

from vllm.model_executor.layers.quantization import modelopt

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPTS = (
    "A ceramic teapot on a wooden table, soft window light, photo",
    "Portrait of an elderly fisherman with a weathered face, golden hour, 85mm",
    "A neon-lit street in Tokyo at night in the rain, reflections on the asphalt",
    "Isometric illustration of a tiny cozy library inside a tree trunk",
    "A poster with the bold headline 'OPEN SOURCE SUMMIT 2026' and a geometric logo",
    "Macro photo of a dewdrop on a green leaf, shallow depth of field",
    "A snowy mountain range under the aurora borealis, ultra wide",
    "Flat vector icon set of kitchen utensils on a white background",
    "An oil painting of a stormy sea with a lighthouse, impasto texture",
    "A bowl of ramen seen from above, food photography, high detail",
    "A futuristic city skyline with flying cars, cinematic matte painting",
    "Hand-drawn pencil sketch of a cat sleeping on a windowsill",
)
SIZES = ((1024, 1024), (768, 1344), (1344, 768), (512, 512))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    amax: dict[str, float] = {}
    original_apply = modelopt.ModelOptLinearMethod.apply

    def recording_apply(self, layer, x, bias=None):
        if getattr(layer, "is_w4a16_nvfp4", False):
            amax[layer.prefix] = max(amax.get(layer.prefix, 0.0), x.abs().amax().item())
        return original_apply(self, layer, x, bias)

    # Calibration only: the in-process diffusion worker picks this up.
    modelopt.ModelOptLinearMethod.apply = recording_apply
    omni = Omni(model=args.model, enforce_eager=True, vae_use_tiling=True)
    for i, prompt in enumerate(PROMPTS):
        height, width = SIZES[i % len(SIZES)]
        params = OmniDiffusionSamplingParams(height=height, width=width, seed=1000 + i, true_cfg_scale=1.0)
        omni.generate({"prompt": prompt}, sampling_params_list=[params])
        print(f"{i + 1}/{len(PROMPTS)} {width}x{height}: {len(amax)} layers seen", flush=True)
    omni.close()

    with open(args.output, "w") as f:
        json.dump(dict(sorted(amax.items())), f, indent=1)


if __name__ == "__main__":
    main()
