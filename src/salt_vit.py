"""SALT / V-JEPA style video ViT (3D RoPE, tubelet 1) run per-frame as an image encoder.

Each frame is encoded alone as a one-frame clip, so the encoder plays the same
role as DINOv2 here: [B, 3, H, W] -> patch tokens [B, H/16 * W/16, C]. With a
single frame the temporal RoPE angle is zero and a frame-level causal mask has
nothing to hide, so the bidirectional and causal students all run through this
same code path.

Module and parameter names mirror the p-salt / jepa-intuitive-physics
``VisionTransformer`` (``use_rope=True``, ``tubelet_size=1``) so its encoder
state dict loads 1:1. RoPE has no positional table and no stored parameters,
so any H, W divisible by the patch size works (224x448 and 448x896 included).
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> (embed_dim, depth, num_heads)
SALT_ARCHS = {
    'salt_vit_small_rope': (384, 12, 6),
    'salt_vit_base_rope': (768, 12, 12),
    'salt_vit_large_rope': (1024, 24, 16),
}


class RoPEAttention3D(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # p-salt layout: three EVEN-width contiguous slices (frame | row | column),
        # leftover channels unrotated. 20 per axis for head_dim 64.
        self.axis_dim = 2 * ((head_dim // 3) // 2)
        omega = 1.0 / 10000 ** (torch.arange(self.axis_dim // 2).float() / (self.axis_dim / 2.0))
        self.register_buffer('omega', omega, persistent=False)

    def rotate(self, x, pos):
        a = self.axis_dim
        sin, cos = [], []
        for p in pos:  # frame, row, column
            f = p[:, None].float() * self.omega[None]          # [N, a/2]
            sin += [f.sin(), f.sin()]                          # TILED [v, v], as in p-salt
            cos += [f.cos(), f.cos()]
        sin, cos = torch.cat(sin, -1), torch.cat(cos, -1)      # [N, 3a]
        xr = x[..., :3 * a].float()
        rot = torch.stack((-xr[..., 1::2], xr[..., 0::2]), dim=-1).flatten(-2)
        return torch.cat(((xr * cos + rot * sin).to(x.dtype), x[..., 3 * a:]), dim=-1)

    def forward(self, x, pos):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(self.rotate(q, pos), self.rotate(k, pos), v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention3D(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        return x + self.mlp(self.norm2(x))


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))

    def forward(self, x):
        # [B, 3, H, W] -> one-frame clip [B, 3, 1, H, W] -> [B, N, C]
        return self.proj(x.unsqueeze(2)).flatten(2).transpose(1, 2)


class SaltViT(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6, patch_size=16, mlp_ratio=4.0):
        super().__init__()
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed3D(patch_size, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward_layers(self, x, layers):
        """Normed patch tokens [B, H*W, C] of the requested blocks (0-based), in block order."""
        p = self.patch_embed.patch_size[0]
        h, w = x.shape[-2] // p, x.shape[-1] // p
        rows, cols = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        rows, cols = rows.flatten(), cols.flatten()
        pos = (torch.zeros_like(rows), rows, cols)  # (frame, row, column) of every token
        x = self.patch_embed(x)
        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, pos)
            if i in layers:
                outs.append(self.norm(x))
        return outs


def build_salt_vit(arch):
    embed_dim, depth, num_heads = SALT_ARCHS[arch]
    return SaltViT(embed_dim=embed_dim, depth=depth, num_heads=num_heads)
