"""SSD-TS denoiser adapted for the unified imputation interface.

Place this file at: model/ssdts/layers.py
Dependencies: mamba_ssm, causal_conv1d, einops, torch.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except Exception as exc:  # pragma: no cover - dependency is optional at import time
    Mamba2 = None
    RMSNorm = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


def _check_mamba_available() -> None:
    if Mamba2 is None or RMSNorm is None:
        raise ImportError(
            "SSD-TS requires mamba_ssm and causal_conv1d. Install versions close to "
            "mamba_ssm==2.2.2 and causal_conv1d==1.4.0, then retry."
        ) from _MAMBA_IMPORT_ERROR


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def flip(seq: torch.Tensor) -> torch.Tensor:
    return torch.flip(seq, dims=[2])


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def cal_diffusion_step_embedding(
    diffusion_steps: torch.Tensor,
    diffusion_step_embed_dim_in: int,
) -> torch.Tensor:
    """Sinusoidal diffusion-step embedding, device-safe version of SSD-TS."""
    assert diffusion_step_embed_dim_in % 2 == 0
    if diffusion_steps.dim() == 1:
        diffusion_steps = diffusion_steps[:, None]
    diffusion_steps = diffusion_steps.float()

    half_dim = diffusion_step_embed_dim_in // 2
    scale = np.log(10000.0) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, device=diffusion_steps.device, dtype=torch.float32) * -scale
    )
    angles = diffusion_steps * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps: int, embedding_dim: int, projection_dim: int | None = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        return F.silu(x)

    @staticmethod
    def _build_embedding(num_steps: int, dim: int = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class MambaEncoderFlip(nn.Module):
    """Bidirectional SSM block along the dimension treated as channels by Conv1d."""

    def __init__(self, in_dim: int, expand: int, headdim: int):
        super().__init__()
        _check_mamba_available()
        self.norm = RMSNorm(in_dim)
        self.input_proj = Conv1d_with_init(in_dim, 2 * in_dim, 1)
        self.weight_proj = Conv1d_with_init(in_dim, 2 * in_dim, 1)
        self.fwd_proj = Conv1d_with_init(2 * in_dim, 2 * in_dim, 1)
        self.fwd_ssm = Mamba2(d_model=2 * in_dim, expand=expand, headdim=headdim)
        self.bwd_proj = Conv1d_with_init(2 * in_dim, 2 * in_dim, 1)
        self.bwd_ssm = Mamba2(d_model=2 * in_dim, expand=expand, headdim=headdim)
        self.out_proj = Conv1d_with_init(2 * in_dim, in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = rearrange(x, "b c l -> b l c")

        ssm_input = self.input_proj(x)
        weight_input = self.weight_proj(x)

        fwd_input = self.fwd_proj(ssm_input)
        fwd_input = rearrange(fwd_input, "b l c -> b c l")
        fwd_input = self.fwd_ssm(fwd_input)
        fwd_input = rearrange(fwd_input, "b c l -> b l c")

        bwd_input = flip(ssm_input)
        bwd_input = self.bwd_proj(bwd_input)
        bwd_input = rearrange(bwd_input, "b l c -> b c l")
        bwd_input = self.bwd_ssm(bwd_input)
        bwd_input = rearrange(bwd_input, "b c l -> b l c")
        bwd_input = flip(bwd_input)

        weight_input = swish(weight_input)
        out = weight_input * fwd_input + weight_input * bwd_input
        out = self.out_proj(out)
        out = rearrange(out, "b l c -> b c l")
        return out + residual


class MambaEncoderForward(nn.Module):
    """Forward SSM block used after transposing channel/time axes."""

    def __init__(self, input_dim: int, expand: int, headdim: int):
        super().__init__()
        _check_mamba_available()
        self.norm = RMSNorm(input_dim)
        self.input_proj = Conv1d_with_init(input_dim, 2 * input_dim, 1)
        self.weight_proj = Conv1d_with_init(input_dim, 2 * input_dim, 1)
        self.fwd_proj = Conv1d_with_init(2 * input_dim, 2 * input_dim, 1)
        self.fwd_ssm = Mamba2(d_model=2 * input_dim, expand=expand, headdim=headdim)
        self.out_proj = Conv1d_with_init(2 * input_dim, input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = rearrange(x, "b c l -> b l c")

        ssm_input = self.input_proj(x)
        ssm_input = self.fwd_proj(ssm_input)
        ssm_input = rearrange(ssm_input, "b l c -> b c l")
        ssm_input = self.fwd_ssm(ssm_input)
        ssm_input = rearrange(ssm_input, "b c l -> b l c")

        weight_input = swish(self.weight_proj(x))
        out = self.out_proj(weight_input * ssm_input)
        out = rearrange(out, "b l c -> b c l")
        return out + residual


class SequentialSSM(nn.Module):
    def __init__(
        self,
        num_ch: int,
        seq_len: int,
        num_ssm: int,
        expand_c: int,
        expand_s: int,
        headdim_c: int,
        headdim_s: int,
    ):
        super().__init__()
        self.num_ssm = num_ssm
        self.ssms = nn.ModuleList()
        for _ in range(num_ssm):
            self.ssms.append(MambaEncoderFlip(seq_len, expand=expand_s, headdim=headdim_s))
            self.ssms.append(MambaEncoderForward(num_ch, expand=expand_c, headdim=headdim_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i in range(0, 2 * self.num_ssm, 2):
            x = self.ssms[i](x)
            x = rearrange(x, "b c l -> b l c")
            x = self.ssms[i + 1](x)
            x = rearrange(x, "b l c -> b c l")
        return residual + x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        diffusion_embedding_dim: int,
        seq_dim: int,
        num_ssm: int,
        num_ch: int,
        seq_len: int,
        cond_ssm_num: int,
        input_ssm_num: int,
        expand_c: int,
        expand_s: int,
        headdim_c: int,
        headdim_s: int,
    ):
        super().__init__()
        self.ssm1 = SequentialSSM(num_ch, seq_dim, num_ssm, expand_c, expand_s, headdim_c, headdim_s)
        self.ssm2 = SequentialSSM(2 * num_ch, seq_dim, num_ssm, expand_c, expand_s, headdim_c, headdim_s)
        self.cond_ssm = SequentialSSM(2 * num_ch, seq_dim, cond_ssm_num, expand_c, expand_s, headdim_c, headdim_s)
        self.input_ssm = SequentialSSM(num_ch, seq_dim, input_ssm_num, expand_c, expand_s, headdim_c, headdim_s)

        self.in_channels = in_channels
        self.res_channels = res_channels
        self.seq_len = seq_len
        self.seq_proj = Conv1d_with_init(seq_len, seq_dim, 1)
        self.diffusion_proj = nn.Linear(diffusion_embedding_dim, res_channels)
        self.input_proj = Conv1d_with_init(in_channels, num_ch, 1)
        self.mid_proj = Conv1d_with_init(res_channels, 2 * res_channels, 1)
        self.res_conv = Conv1d_with_init(res_channels, in_channels, 1)
        self.res_proj_len = Conv1d_with_init(seq_dim, seq_len, 1)
        self.skip_conv = Conv1d_with_init(res_channels, res_channels, 1)
        self.skip_proj_len = Conv1d_with_init(seq_dim, seq_len, 1)
        self.cond_conv = Conv1d_with_init(2 * in_channels, 2 * res_channels, 1)
        self.cond_proj_len = Conv1d_with_init(seq_len, seq_dim, 1)

    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.in_channels, (C, self.in_channels)
        assert L == self.seq_len, (L, self.seq_len)

        h = self.input_proj(h)
        h = rearrange(h, "b c l -> b l c")
        h = self.seq_proj(h)
        h = rearrange(h, "b l c -> b c l")
        h = self.input_ssm(h)

        diffemb = self.diffusion_proj(diffusion_step_embed).view(B, self.res_channels, 1)
        h = h + diffemb
        h = self.ssm1(h)
        h = self.mid_proj(h)

        cond = self.cond_conv(cond)
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.cond_proj_len(cond)
        cond = rearrange(cond, "b l c -> b c l")
        cond = self.cond_ssm(cond)
        h = h + cond

        h = self.ssm2(h)
        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels :, :])

        res = self.res_conv(out)
        res = rearrange(res, "b c l -> b l c")
        res = self.res_proj_len(res)
        res = rearrange(res, "b l c -> b c l")
        assert x.shape == res.shape, (x.shape, res.shape)

        skip = self.skip_conv(out)
        skip = rearrange(skip, "b l c -> b c l")
        skip = self.skip_proj_len(skip)
        skip = rearrange(skip, "b c l -> b l c")
        return (x + res) * math.sqrt(0.5), skip


class BiSSM2Imputer(nn.Module):
    """SSD-TS BiSSM2 denoising network.

    Expected input tuple: (noise_or_x_t, condition, mask, diffusion_step)
    Each tensor except diffusion_step uses shape (B, K, L), where K is num features.
    """

    def __init__(
        self,
        layers: int,
        seq_len: int,
        seq_dim: int,
        in_channels: int,
        res_channels: int,
        diffusion_embedding_dim: int,
        num_steps: int,
        num_ssm: int,
        cond_ssm_num: int,
        input_ssm_num: int,
        num_ch: int,
        expand_c: int,
        expand_s: int,
        headdim_c: int,
        headdim_s: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.diffusion_embedding_dim = diffusion_embedding_dim
        self.layers = layers
        self.diffusion_embedding = DiffusionEmbedding(num_steps=num_steps, embedding_dim=diffusion_embedding_dim)
        self.out_proj1 = Conv1d_with_init(res_channels, in_channels, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels,
                    res_channels=res_channels,
                    diffusion_embedding_dim=diffusion_embedding_dim,
                    seq_dim=seq_dim,
                    cond_ssm_num=cond_ssm_num,
                    input_ssm_num=input_ssm_num,
                    num_ssm=num_ssm,
                    num_ch=num_ch,
                    seq_len=seq_len,
                    expand_c=expand_c,
                    expand_s=expand_s,
                    headdim_c=headdim_c,
                    headdim_s=headdim_s,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        noise, condition, mask, diffusion_step = input_data
        condition = condition * mask
        condition = torch.cat([condition, mask.float()], dim=1)
        diffusion_step_embed = cal_diffusion_step_embedding(diffusion_step, self.diffusion_embedding_dim)

        h = noise
        skip = 0.0
        for layer in self.residual_layers:
            h, skip_n = layer((h, condition, diffusion_step_embed))
            skip = skip + skip_n
        x = skip / math.sqrt(self.layers)
        return self.out_proj1(x)
