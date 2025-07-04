
import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class DiffusionModel(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _p_sample_epsilon(self, x, t, betas, device):
        t = torch.tensor([t]).to(device)
        betas = betas.to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
        sqrt_recip_alphas_t = sqrt_recip_alphas[t]

        model_mean = sqrt_recip_alphas_t * (x - betas[t] * self.forward(x, t) / sqrt_one_minus_alphas_cumprod_t)

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        noise = torch.randn_like(x)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(posterior_variance[t]) * noise
    
    @torch.no_grad()
    def _p_sample_data(self, x, t, betas, device):
        t = torch.tensor([t]).to(device)

        betas = betas.to(device)
        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, axis=0) 
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        model_mean =  (torch.sqrt(alphas[t])*(1-alphas_cumprod_prev[t])*x + torch.sqrt(alphas_cumprod_prev[t])*(1-alphas[t])*self.forward(x, t))/ (1-alphas_cumprod[t])

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        noise = torch.randn_like(x)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(posterior_variance[t]) * noise

    @torch.no_grad()
    def _p_sample_score(self, x, t, betas, device):
        t = torch.tensor([t]).to(device)
        betas = betas.to(device)
        alphas = 1. - betas

        alphas_cumprod = torch.cumprod(alphas, axis=0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t])
        sqrt_recip_alphas_t = sqrt_recip_alphas[t]

        model_mean = sqrt_recip_alphas_t * ( x + betas[t] * self.forward(x, t))

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        noise = torch.randn_like(x)

        if t == 0:
            return model_mean
        else:
            return model_mean + torch.sqrt(posterior_variance[t]) * noise

    @torch.no_grad()
    def _sample_training(self,shape , betas, device, denoising_process_type = "epsilon" ):
        
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)

        for i in reversed(range(len(betas))):
            if denoising_process_type == "noise":
                img = self._p_sample_epsilon(img, i, betas, device)
            elif denoising_process_type == "score":
                img = self._p_sample_score(img, i, betas, device)
            elif denoising_process_type == "data":
                img = self._p_sample_data(img, i, betas, device)
            else:
                raise NotImplementedError("Not implemented for this type of loss choose ['data','noise','score'].")
        return img
    



        
        

        
        