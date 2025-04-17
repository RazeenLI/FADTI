import torch
from torch import nn
import math
import numpy as np

class SeriesDecomposer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride=1,
    ):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def compute_trend(
            self,
            x
    ):
        front = x[:, 0:1, :].repeat(1,12, 1)
        end = x[:, -1:, :].repeat(1, 11, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x

    def forward(
            self,
            x
    ):
        trend = self.compute_trend(x)
        res = x - trend
        return res, trend

class FBMLinear(nn.Module):
    def __init__(
            self,
            context_window,
            target_window,
            dropout=0.1
    ):
        super().__init__()

        sr = context_window
        ts = 1.0/sr
        t = np.arange(0, 1, ts)
        t = torch.tensor(t)
        
        for i in range(context_window//2+1):
            if i == 0:
                cos = 0.5 * torch.cos(2 * math.pi * i * t).unsqueeze(0)
                sin = -0.5 * torch.sin(2 * math.pi * i * t).unsqueeze(0)
            else:
                cos = torch.vstack([cos, torch.cos(2 * math.pi * i * t).unsqueeze(0)])
                sin = torch.vstack([sin, -torch.sin(2 * math.pi * i * t).unsqueeze(0)])

        self.cos = nn.Parameter(cos.float(), requires_grad=False)
        self.sin = nn.Parameter(sin.float(), requires_grad=False)

        linear_input = context_window * (context_window // 2 + 1)

        self.flatten_layer = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(p=dropout)
        self.linear_layer = nn.Linear(linear_input, target_window)

    def forward(
            self,
            x
    ):
        norm = x.size()[-1]
        frequency = torch.fft.rfft(x, axis=-1)
        x = frequency/(norm)*2
        basis_cos=torch.einsum('bkp,pt->bkpt', x.real, self.cos)
        basis_sin=torch.einsum('bkp,pt->bkpt', x.imag, self.sin)

        x = basis_cos + basis_sin
        x = self.flatten_layer(x)
        x = self.dropout(x)
        x = self.linear_layer(x)
        return x
    
class SpectralConv1dLinear(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            context_window, 
            target_window, 
            modes=16, 
            dropout=0.1
        ):
        super().__init__()
        self.modes = modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_window = context_window
        self.target_window = target_window

        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat))
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(context_window, target_window)

    def forward(self, x):
        # x: (B, C, T)
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, F)
        out_ft = torch.zeros(x.size(0), self.out_channels, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum('bci,cio->bio', x_ft[:, :, :self.modes], self.weights)

        x_time = torch.fft.irfft(out_ft, n=self.context_window, dim=-1)  # (B, C_out, T)
        x_time = self.dropout(x_time)
        x_time = x_time.permute(0, 2, 1)  # (B, T, C_out)
        return self.linear(x_time)  # (B, T, target_window)


class FrSSTLike(nn.Module):
    def __init__(self, in_channels, time_steps, num_freqs=64, alpha=0.7, kernel_size=31):
        super().__init__()
        self.num_freqs = num_freqs
        self.time_steps = time_steps
        self.alpha = alpha
        self.kernel_size = kernel_size
        # print("\n\n\n", in_channels, time_steps, num_freqs, alpha, kernel_size, "\n\n\n")

        self.frft_filters = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_freqs,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self._init_frft_filters(alpha)

        self.squeezer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

        self.output_proj = nn.Linear(num_freqs, 1)

    def _init_frft_filters(self, alpha):
        B = torch.arange(self.kernel_size).float()
        center = (self.kernel_size - 1) / 2.0
        t = B - center  # 中心对称

        chirps = []
        for i in range(self.num_freqs):
            omega = (i + 1) * math.pi / self.kernel_size
            real = torch.cos(alpha * t**2 + omega * t).unsqueeze(0)
            chirps.append(real)
        chirps = torch.stack(chirps, dim=0)  # (num_freqs, 1, K)
        self.frft_filters.weight.data = chirps.repeat(self.frft_filters.in_channels, 1, 1)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        x = self.frft_filters(x)               # (B, num_freqs, T)
        x = x.reshape(B, C, self.num_freqs, T)
        x = x * self.squeezer(x)               # frequency reassignment
        x = x.permute(0, 1, 3, 2) # (B, C, T, F)
        x = self.output_proj(x)                # (B, T)
        x = x.squeeze(-1)
        return x


class SpectralReprModule(nn.Module):
    def __init__(
            self, 
            context_window, 
            target_window, 
            num_channels,
            kernel_size=24,
            method="fbm"
        ):
        super().__init__()
        self.decomposer_layer = SeriesDecomposer(kernel_size=kernel_size)

        if method == "frsst":
            self.res_layer = FrSSTLike(
                in_channels=num_channels,
                time_steps=context_window,
            )
            self.trend_layer = FrSSTLike(
                in_channels=num_channels,
                time_steps=context_window,
            )
        elif method == "fsst":
            self.res_layer = FrSSTLike(
                in_channels=num_channels,
                time_steps=context_window,
                alpha=0,
            )
            self.trend_layer = FrSSTLike(
                in_channels=num_channels,
                time_steps=context_window,
                alpha=0
            )
        else:
            self.res_layer = FBMLinear(
                context_window=context_window,
                target_window=target_window
            )
            self.trend_layer = FBMLinear(
                context_window=context_window,
                target_window=target_window
            )

    def forward(self, x):
        res, trend = self.decomposer_layer(x)
        res = res.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        res = self.res_layer(res)
        trend = self.trend_layer(trend)
        return (res + trend).permute(0, 2, 1)


class FourierBasisMapping(nn.Module):
    def __init__(
            self,
            context_window,
            target_window,
            kernel_size=24,
    ):
        super().__init__()

        self.decomposer_layer = SeriesDecomposer(kernel_size=kernel_size)
        self.res_fbm_layer = FBMLinear(
            context_window=context_window,
            target_window=target_window
        )
        self.trend_fbm_layer = FBMLinear(
            context_window=context_window,
            target_window=target_window
        )

    def forward(
            self,
            x,
    ):
        res, trend = self.decomposer_layer(x)
        res, trend = res.permute(0, 2, 1), trend.permute(0, 2, 1)
        res = self.res_fbm_layer(res)
        trend = self.trend_fbm_layer(trend)
        x = res + trend
        x = x.permute(0, 2, 1)

        return x
