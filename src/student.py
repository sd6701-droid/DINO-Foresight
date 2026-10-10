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

A V-JEPA training snapshot holds several models (``encoder``, ``predictor``,
``target_encoder``); append ``::<key>`` to the checkpoint path to choose one,
e.g. ``--student_ckpt best.pt::target_encoder``. Without it the usual key
scan applies, which would pick ``encoder``.

``::<key>`` also accepts a *module prefix* inside a single state dict, for
checkpoints that store several modules side by side (e.g. a Lightning model
with ``frame_encoder.*``, ``temporal_encoder.*``, ...):
``--student_ckpt tdv-small.pth::frame_encoder`` keeps only the keys under
``frame_encoder.`` and strips that prefix. Chunked block names of the DINOv2
``BlockChunk`` form ``blocks.<chunk>.<idx>.*`` are flattened to ``blocks.<idx>.*``.

A timm ``--student_arch`` may carry constructor overrides after a colon,
``name:key=value[,key=value...]`` (values parsed as Python literals), e.g.
``vit_small_patch16_224:patch_size=14`` for a patch-14 ViT-S *without*
LayerScale (the TDV frame encoder). Note ``timm.create_model`` drops kwargs
whose value is ``None``, so a preset default cannot be cleared with ``k=None``;
start from a preset that lacks the feature instead.
"""
import ast
import re

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import checkpoint_filter_fn
from src.salt_vit import SALT_ARCHS, SaltViT, build_salt_vit

DEFAULT_STUDENT_ARCH = 'vit_small_patch14_dinov2'
_STATE_DICT_KEYS = ('state_dict', 'model', 'student', 'encoder', 'teacher', 'model_state_dict', 'module')
_STRIP_PREFIXES = ('module.', 'backbone.', 'student.', 'encoder.', 'model.')


def _unwrap_state_dict(ckpt, key=None):
    """Accept a raw state dict, a Lightning/DINO style wrapper dict, or a pickled nn.Module.

    ``key`` names the sub-dict to load explicitly (e.g. ``target_encoder`` of a
    V-JEPA training snapshot, which also holds ``encoder`` and ``predictor``).
    Without it the first match in _STATE_DICT_KEYS wins, as before.
    Returns ``(state_dict, selected)`` where ``selected`` is the sub-dict key used
    (None for a raw state dict) so the caller can report it.
    """
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict(), None
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type {type(ckpt)}; expected a state dict or nn.Module")
    selected = None
    if key is not None and key in ckpt and isinstance(ckpt[key], dict):
        ckpt, selected = ckpt[key], key
    else:
        for k in _STATE_DICT_KEYS:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt, selected = ckpt[k], k
                break
    state_dict = {}
    for k, v in ckpt.items():
        for prefix in _STRIP_PREFIXES:
            if k.startswith(prefix):
                k = k[len(prefix):]
        state_dict[k] = v
    if key is not None and selected != key:
        # Not a top-level sub-dict: treat ``key`` as a module prefix inside the state dict
        # (e.g. ``frame_encoder`` of a Lightning model that also stores other modules).
        # Applied after the generic wrapper prefixes (``model.``, ``module.`` ...) are stripped.
        prefix = key.rstrip('.') + '.'
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sub:
            raise KeyError(
                f"Checkpoint has neither a state-dict entry nor a module prefix '{key}'; "
                f"top-level keys: {list(ckpt.keys())[:10]}; "
                f"module prefixes: {sorted({k.split('.')[0] for k in state_dict})}")
        state_dict, selected = sub, (f"{selected}/{key}" if selected else key)
    return _flatten_chunked_blocks(state_dict), selected


_CHUNKED_BLOCK_RE = re.compile(r'^blocks\.(\d+)\.(\d+)\.(.+)$')


def _flatten_chunked_blocks(state_dict):
    """Map ``blocks.<chunk>.<idx>.*`` to ``blocks.<n>.*`` so chunked ViTs load into a flat timm ViT.

    DINOv2's ``BlockChunk`` keeps the GLOBAL block index as ``<idx>`` (chunk 1 holds
    ``blocks.1.3``, ``blocks.1.4``, ...), so when the inner indices are all distinct they are
    used directly. Otherwise (plain nested lists restarting at 0 per chunk) blocks are numbered
    in (chunk, idx) order. Keys that are not of this form are returned unchanged.
    """
    matches = {k: m for k in state_dict for m in [_CHUNKED_BLOCK_RE.match(k)] if m}
    if not matches:
        return state_dict
    pairs = sorted({(int(m.group(1)), int(m.group(2))) for m in matches.values()})
    inner = [i for _, i in pairs]
    if len(set(inner)) == len(inner):
        index = {pair: pair[1] for pair in pairs}
    else:
        index = {pair: n for n, pair in enumerate(pairs)}
    out = {}
    for k, v in state_dict.items():
        m = matches.get(k)
        if m is not None:
            k = f"blocks.{index[(int(m.group(1)), int(m.group(2)))]}.{m.group(3)}"
        out[k] = v
    return out


def _parse_arch(arch):
    """Split ``name[:k=v,k=v...]`` into the timm model name and constructor kwargs.

    Values go through ``ast.literal_eval`` (numbers, bools, tuples), falling back to the
    raw string. ``vit_small_patch16_224:patch_size=14`` -> patch-14 ViT-S, no LayerScale.
    ``None`` values are discarded by ``timm.create_model`` and therefore cannot override.
    """
    name, _, spec = arch.partition(':')
    overrides = {}
    for item in filter(None, spec.split(',')):
        k, sep, v = item.partition('=')
        if not sep or not k.strip():
            raise ValueError(f"Bad --student_arch override '{item}' in '{arch}'; expected key=value")
        try:
            overrides[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            overrides[k.strip()] = v.strip()
    return name.strip(), overrides


def load_student(ckpt_path, arch=DEFAULT_STUDENT_ARCH, img_size=(448, 896), verbose=True):
    """Build a frozen timm ViT and load the student weights from ``ckpt_path``.

    ``dynamic_img_size=True`` lets the same weights run at 224x448 and 448x896
    (positional embeddings are interpolated on the fly), which the two-stage
    training relies on.

    ``ckpt_path`` may be ``path::key`` to pick one sub-dict of a multi-model
    snapshot, e.g. ``best.pt::target_encoder`` for the EMA target encoder of a
    V-JEPA run (whose ``encoder`` would otherwise be picked up first).
    """
    if ckpt_path is None:
        raise ValueError("--feature_extractor student requires --student_ckpt /path/to/student.pth")
    path, _, key = ckpt_path.partition('::')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict, selected = _unwrap_state_dict(ckpt, key or None)
    if arch in SALT_ARCHS:
        model = build_salt_vit(arch)
        # RoPE has no stored parameters (omega is a non-persistent buffer),
        # so every checkpoint key should map 1:1 to a model parameter.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        name, overrides = _parse_arch(arch)
        model = timm.create_model(name, pretrained=False, img_size=tuple(img_size),
                                  dynamic_img_size=True, num_classes=0, **overrides)
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
        print(f"[student] Loaded {arch} from {ckpt_path} (sub-dict={selected}, embed_dim={model.embed_dim}, "
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
