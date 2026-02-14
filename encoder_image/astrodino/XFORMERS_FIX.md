# xFormers CUDA Warp Alignment Fix

## Problem

During DINOv2 training, the following error occurs:

```
RuntimeError: Expected source.size(-1) % kNumElementsPerWarp == 0 to be true, but got false.
```

This error originates from xformers' `scaled_index_add` operation during the forward pass of the vision transformer blocks.

## Root Cause

The error occurs when the sequence length (number of tokens) in the vision transformer is not properly aligned with CUDA warp size requirements. xformers' CUDA kernels require that tensor dimensions be divisible by specific values (typically 8) to ensure efficient warp-level operations.

In DINOv2 vision transformers, the sequence length is calculated as:
```
sequence_length = (image_size / patch_size)^2 + 1 (CLS token)
```

When this sequence length is not divisible by 8, xformers' CUDA operations fail.

## Solution

Add **register tokens** to pad the sequence length to be divisible by 8. Register tokens are additional learnable tokens that don't affect the core functionality but ensure proper alignment.

### Configuration Changes

| Config File | Architecture | Calculation | Register Tokens |
|-------------|--------------|-------------|-----------------|
| config_default.yaml | vit_base (patch=12, size=144) | (144/12)^2 + 1 = 145 | **7** → 152 |
| astrodino.yaml | vit_large (patch=12, size=144) | (144/12)^2 + 1 = 145 | **7** → 152 |
| astrodino_desi_vitb.yaml | vit_base (patch=12, size=144) | (144/12)^2 + 1 = 145 | **7** → 152 |
| astrodino_desi_vitl.yaml | vit_large (patch=12, size=144) | (144/12)^2 + 1 = 145 | **7** → 152 |
| astrodino_jwst_vitl_f115w.yaml | vit_large (patch=10, size=90) | (90/10)^2 + 1 = 82 | **6** → 88 |
| astrodino_jwst_vitb_f150w.yaml | vit_small (patch=8, size=64) | (64/8)^2 + 1 = 65 | **7** → 72 |

### How to Calculate Required Register Tokens

1. Calculate base tokens: `tokens = (image_size / patch_size)^2 + 1`
2. Calculate remainder: `remainder = tokens % 8`
3. If remainder != 0, add register tokens: `num_register_tokens = 8 - remainder`

Example Python code:
```python
patch_size = 12
global_crops_size = 144
tokens = (global_crops_size // patch_size) ** 2 + 1
num_register_tokens = (8 - tokens % 8) % 8
print(f"Tokens: {tokens}, Register tokens needed: {num_register_tokens}")
```

## Implementation

Add the following line to the `student` section of your config file:

```yaml
student:
  arch: vit_base  # or vit_small, vit_large
  patch_size: 12
  # ... other parameters ...
  num_register_tokens: 7  # Fix xFormers CUDA warp alignment
```

## References

- DINOv2: https://github.com/facebookresearch/dinov2
- xFormers: https://github.com/facebookresearch/xformers
- Register tokens in vision transformers: Similar to positional embeddings, these are learnable parameters that help with model capacity and now serve the additional purpose of ensuring proper CUDA alignment.
