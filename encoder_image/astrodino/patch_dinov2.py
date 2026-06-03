#!/usr/bin/env python3
"""Idempotently patch the installed dinov2 package with project-specific modifications.

Patches applied:
  models/__init__.py     — build_model: accept in_chans/channel/embed_dim/depth/num_heads/mlp_ratio
  models/vision_transformer.py — vit_{small,base,large,giant2}: allow kwargs to override defaults
"""

import os
import sys


def _dinov2_dir() -> str:
    try:
        import dinov2
        return os.path.dirname(dinov2.__file__)
    except ImportError:
        sys.exit("dinov2 is not installed; run `pixi run install-dinov2` first")


def patch_file(path: str, old: str, new: str, label: str) -> None:
    with open(path) as f:
        content = f.read()
    if new in content:
        print(f"  [skip] {label} — already patched")
        return
    if old not in content:
        print(f"  [WARN] {label} — expected pattern not found, skipping", file=sys.stderr)
        return
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    print(f"  [ok]   {label}")


def patch_init(dinov2_dir: str) -> None:
    path = os.path.join(dinov2_dir, "models", "__init__.py")

    old = (
        "        vit_kwargs = dict(\n"
        "            img_size=img_size,\n"
        "            patch_size=args.patch_size,\n"
        "            init_values=args.layerscale,\n"
        "            ffn_layer=args.ffn_layer,\n"
        "            block_chunks=args.block_chunks,\n"
        "            qkv_bias=args.qkv_bias,\n"
        "            proj_bias=args.proj_bias,\n"
        "            ffn_bias=args.ffn_bias,\n"
        "            num_register_tokens=args.num_register_tokens,\n"
        "            interpolate_offset=args.interpolate_offset,\n"
        "            interpolate_antialias=args.interpolate_antialias,\n"
        "        )\n"
        "        teacher = vits.__dict__[args.arch](**vit_kwargs)"
    )

    new = (
        "        vit_kwargs = dict(\n"
        "            img_size=img_size,\n"
        "            patch_size=args.patch_size,\n"
        "            init_values=args.layerscale,\n"
        "            ffn_layer=args.ffn_layer,\n"
        "            block_chunks=args.block_chunks,\n"
        "            qkv_bias=args.qkv_bias,\n"
        "            proj_bias=args.proj_bias,\n"
        "            ffn_bias=args.ffn_bias,\n"
        "            num_register_tokens=args.num_register_tokens,\n"
        "            interpolate_offset=args.interpolate_offset,\n"
        "            interpolate_antialias=args.interpolate_antialias,\n"
        "            in_chans=args.get('in_chans', 3),\n"
        "            channel=args.get('channel', None),\n"
        "            # Allow overriding model architecture params\n"
        "            embed_dim=args.get('embed_dim', None),\n"
        "            depth=args.get('depth', None),\n"
        "            num_heads=args.get('num_heads', None),\n"
        "            mlp_ratio=args.get('mlp_ratio', None),\n"
        "            patch_stride=args.get('patch_stride', None),\n"
        "        )\n"
        "        # Remove None values so they use function defaults\n"
        "        vit_kwargs = {k: v for k, v in vit_kwargs.items() if v is not None}\n"
        "        teacher = vits.__dict__[args.arch](**vit_kwargs)"
    )

    patch_file(path, old, new, "models/__init__.py :: build_model")

    # Incremental: file was already patched (has in_chans etc.) but is missing patch_stride.
    # patch_file is idempotent: skips if new already present, skips if old already absent.
    patch_file(
        path,
        "            in_chans=args.get('in_chans', 3),\n"
        "            channel=args.get('channel', None),\n"
        "            # Allow overriding model architecture params\n"
        "            embed_dim=args.get('embed_dim', None),\n"
        "            depth=args.get('depth', None),\n"
        "            num_heads=args.get('num_heads', None),\n"
        "            mlp_ratio=args.get('mlp_ratio', None),\n"
        "        )\n"
        "        # Remove None values so they use function defaults\n",
        "            in_chans=args.get('in_chans', 3),\n"
        "            channel=args.get('channel', None),\n"
        "            # Allow overriding model architecture params\n"
        "            embed_dim=args.get('embed_dim', None),\n"
        "            depth=args.get('depth', None),\n"
        "            num_heads=args.get('num_heads', None),\n"
        "            mlp_ratio=args.get('mlp_ratio', None),\n"
        "            patch_stride=args.get('patch_stride', None),\n"
        "        )\n"
        "        # Remove None values so they use function defaults\n",
        "models/__init__.py :: build_model patch_stride (incremental)",
    )


def _vit_patch(name: str, defaults: dict) -> tuple[str, str]:
    """Build (old, new) patch strings for a vit_* factory function."""
    embed_dim = defaults["embed_dim"]
    depth     = defaults["depth"]
    num_heads = defaults["num_heads"]
    mlp_ratio = defaults["mlp_ratio"]
    docstring = defaults.get("docstring", "")

    doc_block = f'    """\n{docstring}    """\n' if docstring else ""

    old = (
        f"def {name}(patch_size=16, num_register_tokens=0, **kwargs):\n"
        + doc_block
        + "    model = DinoVisionTransformer(\n"
        f"        patch_size=patch_size,\n"
        f"        embed_dim={embed_dim},\n"
        f"        depth={depth},\n"
        f"        num_heads={num_heads},\n"
        f"        mlp_ratio={mlp_ratio},\n"
        "        block_fn=partial(Block, attn_class=MemEffAttention),\n"
        "        num_register_tokens=num_register_tokens,\n"
        "        **kwargs,\n"
        "    )\n"
        "    return model"
    )

    new = (
        f"def {name}(patch_size=16, num_register_tokens=0, **kwargs):\n"
        + doc_block
        + "    # Allow overriding default values from kwargs\n"
        f"    embed_dim = kwargs.pop('embed_dim', {embed_dim})\n"
        f"    depth = kwargs.pop('depth', {depth})\n"
        f"    num_heads = kwargs.pop('num_heads', {num_heads})\n"
        f"    mlp_ratio = kwargs.pop('mlp_ratio', {mlp_ratio})\n"
        "    model = DinoVisionTransformer(\n"
        "        patch_size=patch_size,\n"
        "        embed_dim=embed_dim,\n"
        "        depth=depth,\n"
        "        num_heads=num_heads,\n"
        "        mlp_ratio=mlp_ratio,\n"
        "        block_fn=partial(Block, attn_class=MemEffAttention),\n"
        "        num_register_tokens=num_register_tokens,\n"
        "        **kwargs,\n"
        "    )\n"
        "    return model"
    )

    return old, new


def patch_vision_transformer(dinov2_dir: str) -> None:
    path = os.path.join(dinov2_dir, "models", "vision_transformer.py")

    vit_variants = [
        ("vit_small",  {"embed_dim": 384,  "depth": 12, "num_heads": 6,  "mlp_ratio": 4}),
        ("vit_base",   {"embed_dim": 768,  "depth": 12, "num_heads": 12, "mlp_ratio": 4}),
        ("vit_large",  {"embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4}),
        ("vit_giant2", {"embed_dim": 1536, "depth": 40, "num_heads": 24, "mlp_ratio": 4,
                        "docstring": "    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64\n"}),
    ]

    for name, defaults in vit_variants:
        old, new = _vit_patch(name, defaults)
        patch_file(path, old, new, f"models/vision_transformer.py :: {name}")


def patch_patch_embed(dinov2_dir: str) -> None:
    path = os.path.join(dinov2_dir, "layers", "patch_embed.py")

    patch_file(
        path,
        "    def __init__(\n"
        "        self,\n"
        "        img_size: Union[int, Tuple[int, int]] = 224,\n"
        "        patch_size: Union[int, Tuple[int, int]] = 16,\n"
        "        in_chans: int = 3,\n"
        "        embed_dim: int = 768,\n"
        "        norm_layer: Optional[Callable] = None,\n"
        "        flatten_embedding: bool = True,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "\n"
        "        image_HW = make_2tuple(img_size)\n"
        "        patch_HW = make_2tuple(patch_size)\n"
        "        patch_grid_size = (\n"
        "            image_HW[0] // patch_HW[0],\n"
        "            image_HW[1] // patch_HW[1],\n"
        "        )\n"
        "\n"
        "        self.img_size = image_HW\n"
        "        self.patch_size = patch_HW\n"
        "        self.patches_resolution = patch_grid_size\n"
        "        self.num_patches = patch_grid_size[0] * patch_grid_size[1]\n"
        "\n"
        "        self.in_chans = in_chans\n"
        "        self.embed_dim = embed_dim\n"
        "\n"
        "        self.flatten_embedding = flatten_embedding\n"
        "\n"
        "        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)",
        "    def __init__(\n"
        "        self,\n"
        "        img_size: Union[int, Tuple[int, int]] = 224,\n"
        "        patch_size: Union[int, Tuple[int, int]] = 16,\n"
        "        in_chans: int = 3,\n"
        "        embed_dim: int = 768,\n"
        "        norm_layer: Optional[Callable] = None,\n"
        "        flatten_embedding: bool = True,\n"
        "        stride: Union[int, Tuple[int, int], None] = None,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "\n"
        "        image_HW = make_2tuple(img_size)\n"
        "        patch_HW = make_2tuple(patch_size)\n"
        "        stride_HW = make_2tuple(stride) if stride is not None else patch_HW\n"
        "        patch_grid_size = (\n"
        "            (image_HW[0] - patch_HW[0]) // stride_HW[0] + 1,\n"
        "            (image_HW[1] - patch_HW[1]) // stride_HW[1] + 1,\n"
        "        )\n"
        "\n"
        "        self.img_size = image_HW\n"
        "        self.patch_size = patch_HW\n"
        "        self.stride = stride_HW\n"
        "        self.patches_resolution = patch_grid_size\n"
        "        self.num_patches = patch_grid_size[0] * patch_grid_size[1]\n"
        "\n"
        "        self.in_chans = in_chans\n"
        "        self.embed_dim = embed_dim\n"
        "\n"
        "        self.flatten_embedding = flatten_embedding\n"
        "\n"
        "        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=stride_HW)",
        "layers/patch_embed.py :: PatchEmbed.__init__ stride",
    )

    patch_file(
        path,
        '        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"\n'
        '        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"',
        '        assert H >= patch_H, f"Input image height {H} is smaller than patch height {patch_H}"\n'
        '        assert W >= patch_W, f"Input image width {W} is smaller than patch width {patch_W}"',
        "layers/patch_embed.py :: PatchEmbed.forward assertions",
    )


def patch_vision_transformer_stride(dinov2_dir: str) -> None:
    path = os.path.join(dinov2_dir, "models", "vision_transformer.py")

    # Spot A: add patch_stride param to __init__ and wire up
    patch_file(
        path,
        "        interpolate_offset=0.1,\n"
        "    ):\n"
        "        \"\"\"\n"
        "        Args:\n"
        "            img_size (int, tuple): input image size\n"
        "            patch_size (int, tuple): patch size\n"
        "            in_chans (int): number of input channels\n"
        "            embed_dim (int): embedding dimension\n"
        "            depth (int): depth of transformer\n"
        "            num_heads (int): number of attention heads\n"
        "            mlp_ratio (int): ratio of mlp hidden dim to embedding dim\n"
        "            qkv_bias (bool): enable bias for qkv if True\n"
        "            proj_bias (bool): enable bias for proj in attn if True\n"
        "            ffn_bias (bool): enable bias for ffn if True\n"
        "            drop_path_rate (float): stochastic depth rate\n"
        "            drop_path_uniform (bool): apply uniform drop rate across blocks\n"
        "            weight_init (str): weight init scheme\n"
        "            init_values (float): layer-scale init values\n"
        "            embed_layer (nn.Module): patch embedding layer\n"
        "            act_layer (nn.Module): MLP activation layer\n"
        "            block_fn (nn.Module): transformer block class\n"
        "            ffn_layer (str): \"mlp\", \"swiglu\", \"swiglufused\" or \"identity\"\n"
        "            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap\n"
        "            num_register_tokens: (int) number of extra cls tokens (so-called \"registers\")\n"
        "            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings\n"
        "            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings\n"
        "        \"\"\"\n"
        "        super().__init__()\n"
        "        norm_layer = partial(nn.LayerNorm, eps=1e-6)\n"
        "\n"
        "        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models\n"
        "        self.num_tokens = 1\n"
        "        self.n_blocks = depth\n"
        "        self.num_heads = num_heads\n"
        "        self.patch_size = patch_size\n"
        "        self.num_register_tokens = num_register_tokens\n"
        "        self.interpolate_antialias = interpolate_antialias\n"
        "        self.interpolate_offset = interpolate_offset\n"
        "\n"
        "        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)",
        "        interpolate_offset=0.1,\n"
        "        patch_stride=None,\n"
        "    ):\n"
        "        \"\"\"\n"
        "        Args:\n"
        "            img_size (int, tuple): input image size\n"
        "            patch_size (int, tuple): patch size\n"
        "            in_chans (int): number of input channels\n"
        "            embed_dim (int): embedding dimension\n"
        "            depth (int): depth of transformer\n"
        "            num_heads (int): number of attention heads\n"
        "            mlp_ratio (int): ratio of mlp hidden dim to embedding dim\n"
        "            qkv_bias (bool): enable bias for qkv if True\n"
        "            proj_bias (bool): enable bias for proj in attn if True\n"
        "            ffn_bias (bool): enable bias for ffn if True\n"
        "            drop_path_rate (float): stochastic depth rate\n"
        "            drop_path_uniform (bool): apply uniform drop rate across blocks\n"
        "            weight_init (str): weight init scheme\n"
        "            init_values (float): layer-scale init values\n"
        "            embed_layer (nn.Module): patch embedding layer\n"
        "            act_layer (nn.Module): MLP activation layer\n"
        "            block_fn (nn.Module): transformer block class\n"
        "            ffn_layer (str): \"mlp\", \"swiglu\", \"swiglufused\" or \"identity\"\n"
        "            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap\n"
        "            num_register_tokens: (int) number of extra cls tokens (so-called \"registers\")\n"
        "            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings\n"
        "            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings\n"
        "        \"\"\"\n"
        "        super().__init__()\n"
        "        norm_layer = partial(nn.LayerNorm, eps=1e-6)\n"
        "\n"
        "        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models\n"
        "        self.num_tokens = 1\n"
        "        self.n_blocks = depth\n"
        "        self.num_heads = num_heads\n"
        "        self.patch_size = patch_size\n"
        "        self.patch_stride = patch_stride if patch_stride is not None else patch_size\n"
        "        self.num_register_tokens = num_register_tokens\n"
        "        self.interpolate_antialias = interpolate_antialias\n"
        "        self.interpolate_offset = interpolate_offset\n"
        "\n"
        "        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_stride)",
        "models/vision_transformer.py :: DinoVisionTransformer.__init__ patch_stride",
    )

    # Spot B: interpolate_pos_encoding grid calculation
    patch_file(
        path,
        "        w0 = w // self.patch_size\n"
        "        h0 = h // self.patch_size",
        "        w0 = (w - self.patch_size) // self.patch_stride + 1\n"
        "        h0 = (h - self.patch_size) // self.patch_stride + 1",
        "models/vision_transformer.py :: interpolate_pos_encoding grid",
    )

    # Spot C: get_intermediate_layers reshape
    patch_file(
        path,
        "                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()",
        "                out.reshape(B, (w - self.patch_size) // self.patch_stride + 1, (h - self.patch_size) // self.patch_stride + 1, -1).permute(0, 3, 1, 2).contiguous()",
        "models/vision_transformer.py :: get_intermediate_layers reshape",
    )


def patch_fsdp(dinov2_dir: str) -> None:
    path = os.path.join(dinov2_dir, "fsdp", "__init__.py")

    # save(): use FULL_STATE_DICT so all ranks write a single consolidated checkpoint
    patch_file(
        path,
        "        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):\n"
        "            data[\"model\"] = self.model.state_dict()",
        "        #with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):\n"
        "        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):\n"
        "            data[\"model\"] = self.model.state_dict()",
        "fsdp/__init__.py :: FSDPCheckpointer.save",
    )

    # load(): match FULL_STATE_DICT so loading works with the consolidated checkpoint
    patch_file(
        path,
        "        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):\n"
        "            return super().load(*args, **kwargs)",
        "        #with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):\n"
        "        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):\n"
        "            return super().load(*args, **kwargs)",
        "fsdp/__init__.py :: FSDPCheckpointer.load",
    )


def main() -> None:
    dinov2_dir = _dinov2_dir()
    print(f"dinov2 @ {dinov2_dir}")
    patch_init(dinov2_dir)
    patch_vision_transformer(dinov2_dir)
    patch_patch_embed(dinov2_dir)
    patch_vision_transformer_stride(dinov2_dir)
    patch_fsdp(dinov2_dir)
    print("done")


if __name__ == "__main__":
    main()
