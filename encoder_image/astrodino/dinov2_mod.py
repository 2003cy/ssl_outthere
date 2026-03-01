#the following patches should be replaaced to the dinov2 codebase, for more flexibility in setting up ViT with fully customizable architecture parameters, and for better handling of pretrained weights with different key names. These changes are needed for our experiments with custom ViT architectures and pretrained weights.

#apply the following to: dinov2/models/__init__.py
#-----------existing code in __init__.py----------------
def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            in_chans=args.get('in_chans', 3),
            channel=args.get('channel', None),
            # Allow overriding model architecture params
            embed_dim=args.get('embed_dim', None),
            depth=args.get('depth', None),
            num_heads=args.get('num_heads', None),
            mlp_ratio=args.get('mlp_ratio', None),
        )
        # Remove None values so they use function defaults
        vit_kwargs = {k: v for k, v in vit_kwargs.items() if v is not None}
        
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

#-----------new code in __init__.py----------------



#apply the following to: dinov2/models/vision_transformer.py
#-----------existing code in vision_transformer.py----------------
def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    # Allow overriding default values from kwargs
    embed_dim = kwargs.pop('embed_dim', 384)
    depth = kwargs.pop('depth', 12)
    num_heads = kwargs.pop('num_heads', 6)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    # Allow overriding default values from kwargs
    embed_dim = kwargs.pop('embed_dim', 768)
    depth = kwargs.pop('depth', 12)
    num_heads = kwargs.pop('num_heads', 12)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    # Allow overriding default values from kwargs
    embed_dim = kwargs.pop('embed_dim', 1024)
    depth = kwargs.pop('depth', 24)
    num_heads = kwargs.pop('num_heads', 16)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    # Allow overriding default values from kwargs
    embed_dim = kwargs.pop('embed_dim', 1536)
    depth = kwargs.pop('depth', 40)
    num_heads = kwargs.pop('num_heads', 24)
    mlp_ratio = kwargs.pop('mlp_ratio', 4)
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
#-----------new code in vision_transformer.py----------------
