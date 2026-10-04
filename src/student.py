"""Frozen student encoder (loaded from a .pth) as a drop-in feature extractor.

Used by train.py (src/dino_f.py), pca.py and Downstream/dinov2dpt.py through
``--feature_extractor student --student_ckpt /path/to/student.pth``.

The encoder is built with timm so that ``forward_intermediates`` gives us the
per-layer patch tokens in [B, N, C] format, exactly like DINOv2's
``get_intermediate_layers``. Pick the architecture with ``--student_arch``
(any timm VisionTransformer name). The default ``vit_small_patch14_dinov2`` is
a ViT-S/14 with LayerScale and no register tokens.

``--student_arch salt_vit_small_rope`` (see src/salt_vit.py) instead loads a
SALT / V-JEPA style video ViT-S/16 and runs it per-frame.
"""
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from src.salt_vit import SALT_ARCHS, SaltViT, build_salt_vit

DEFAULT_STUDENT_ARCH = 'vit_small_patch14_dinov2'
_STATE_DICT_KEYS = ('state_dict', 'model', 'student', 'encoder', 'teacher', 'model_state_dict', 'module')
_STRIP_PREFIXES = ('module.', 'backbone.', 'student.', 'encoder.', 'model.')


def _unwrap_state_dict(ckpt):
    """Accept a raw state dict, a Lightning/DINO style wrapper dict, or a pickled nn.Module."""
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type {type(ckpt)}; expected a state dict or nn.Module")
    for key in _STATE_DICT_KEYS:
        if key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    state_dict = {}
    for k, v in ckpt.items():
        for prefix in _STRIP_PREFIXES:
            if k.startswith(prefix):
                k = k[len(prefix):]
        state_dict[k] = v
    return state_dict


def load_student(ckpt_path, arch=DEFAULT_STUDENT_ARCH, img_size=(448, 896), verbose=True):
    """Build a frozen timm ViT and load the student weights from ``ckpt_path``.

    ``dynamic_img_size=True`` lets the same weights run at 224x448 and 448x896
    (positional embeddings are interpolated on the fly), which the two-stage
    training relies on.
    """
    if ckpt_path is None:
        raise ValueError("--feature_extractor student requires --student_ckpt /path/to/student.pth")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = _unwrap_state_dict(ckpt)
    if arch in SALT_ARCHS:
        model = build_salt_vit(arch)
        # RoPE has no stored parameters (omega is a non-persistent buffer),
        # so every checkpoint key should map 1:1 to a model parameter.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        model = timm.create_model(arch, pretrained=False, img_size=tuple(img_size),
                                  dynamic_img_size=True, num_classes=0)
        # Resamples pos_embed to the model grid and adapts key names of common ViT formats.
        state_dict = checkpoint_filter_fn(state_dict, model)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # The classifier head is removed (num_classes=0), so head.* keys are expected to be unused.
        unexpected = [k for k in unexpected if not k.startswith('head')]
    if missing:
        raise RuntimeError(
            f"Student checkpoint {ckpt_path} does not match architecture '{arch}'. "
            f"Missing keys: {missing}. Choose the matching model name with --student_arch.")
    if verbose and unexpected:
        print(f"[student] Ignoring unexpected keys in {ckpt_path}: {unexpected}")
    if verbose:
        print(f"[student] Loaded {arch} from {ckpt_path} (embed_dim={model.embed_dim}, "
              f"patch_size={model.patch_embed.patch_size[0]}, depth={len(model.blocks)})")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def student_intermediates(model, x, layers):
    """Patch tokens of the requested blocks (0-based), each of shape [B, H*W, C], cls/register tokens dropped."""
    layers = [layers] if isinstance(layers, int) else list(layers)
    with torch.no_grad():
        if isinstance(model, SaltViT):
            return model.forward_layers(x, layers)
        return model.forward_intermediates(x, indices=layers, output_fmt='NLC', norm=True,
                                           intermediates_only=True)
