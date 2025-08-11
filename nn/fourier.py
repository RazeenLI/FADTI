import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F


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

class DummyExtension(nn.Module):
    def __init__(self, context_window, target_window):
        super().__init__()
        self.context_window = context_window
        self.target_window = target_window

    def forward(self, x):
        """
        x: [B, C, context_window]
        output: [B, C, target_window]
        不对输入内容进行任何变换，只进行裁剪或 padding
        """
        T = x.size(-1)
        if T > self.target_window:
            return x[..., :self.target_window]  # 裁剪右边
        elif T < self.target_window:
            pad_len = self.target_window - T
            return F.pad(x, (0, pad_len))  # 右侧 padding 0
        else:
            return x
        

class DFTBasisExtension(nn.Module):
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
        basis_cos=torch.einsum('bcf,ft->bcft', x.real, self.cos)
        basis_sin=torch.einsum('bcf,ft->bcft', x.imag, self.sin)

        x = basis_cos + basis_sin
        x = self.flatten_layer(x)
        x = self.dropout(x)
        x = self.linear_layer(x)
        return x
    
class STFTBiasProjection(nn.Module):
    def __init__(
            self,
            context_window,
            target_window,
            dropout=0.1
    ):
        super().__init__()
        self.context_window = context_window
        self.hop_length = context_window // 2
        # 1）预先生成一个窗函数
        self.register_buffer('window', torch.hann_window(context_window))

        # 2）对应 STFT 得到的频率 bin 数
        freq_bins = context_window // 2 + 1  
        # 线性层输入维度 = freq_bins × frame_count
        frame_count = ((target_window or context_window) - context_window) // self.hop_length + 1

        N = torch.arange(frame_count).float()  # time frame indices
        omega = 2 * math.pi * torch.arange(freq_bins).float().unsqueeze(1) / frame_count  # angular frequency matrix
        cos = torch.cos(omega * N)
        sin = -torch.sin(omega * N)
        cos[0] *= 0.5  # match DFT basis
        sin[0] *= 0.5

        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

        linear_input = freq_bins * frame_count
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(linear_input, target_window)

    def forward(self, x):
        # x: (B, C, T), 最后一维是 time
        # —— 把 DFT 换成 STFT —— 
        # x: (B, C, T)
        B, C, T = x.shape
        # 1) STFT per channel -> (B, C, freq_bins, frames)
        x2d = x.reshape(B*C, T)
        X_stft = torch.stft(
            x2d,
            n_fft=self.context_window,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=False
        ) # (B*C, freq_bins, frames)
        # reshape (B, C, F, L)
        energy = self.window.pow(2).sum()
        X_stft = X_stft / energy  # normalize by window energy

        freq_bins, frames = X_stft.size(1), X_stft.size(2)
        X_stft = X_stft.reshape(B, C, freq_bins, frames)

        # 保持原来基于实部/虚部的扩展逻辑（可选）
        basis_cos = torch.einsum('bcfl,fl->bcfl', X_stft.real, self.cos) # X_stft.real # (B, C, freq_bins, frames) real
        basis_sin = torch.einsum('bcfl,fl->bcfl', X_stft.imag, self.sin) # X_stft.imag # imag
        x = basis_cos + basis_sin

        # —— 扁平 & 投影 —— 
        x = self.flatten(x)   # (B, C, freq_bins*frames*2)
        x = self.dropout(x)
        x = self.linear(x)    # (B, C, target_window)
        return x

class FrSSTBiasProjection(nn.Module):
    """
    Fourier Synchrosqueezed Transform (FrSST) bias projection for univariate signals.
    """
    def __init__(
            self, 
            context_window, 
            target_window, 
            dropout=0.1
    ):
        super().__init__()
        self.context_window = context_window
        self.hop_length = context_window // 2
        # Hann window for STFT
        self.register_buffer('window', torch.hann_window(self.context_window))
        
        # Number of frequency bins in STFT output
        self.freq_bins = self.context_window // 2 + 1
        frame_count = ((target_window or context_window) - context_window) // self.hop_length + 1

        N = torch.arange(frame_count).float()  # time frame indices
        omega = 2 * math.pi * torch.arange(self.freq_bins).float().unsqueeze(1) / frame_count  # angular frequency matrix
        cos = torch.cos(omega * N)
        sin = -torch.sin(omega * N)
        cos[0] *= 0.5  # match DFT basis
        sin[0] *= 0.5

        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
        # Linear mapping from F bins to target_window
        self.flatten = nn.Flatten(start_dim=-2)  # flatten C and freq_bins
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.freq_bins * frame_count, target_window)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # 1) STFT per channel -> (B, C, freq_bins, frames)
        x2d = x.reshape(B*C, T)
        X_stft = torch.stft(
            x2d,
            n_fft=self.context_window,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=False
        ) # (B*C, freq_bins, frames)
        # reshape (B, C, F, L)
        energy = self.window.pow(2).sum()
        X_stft = X_stft / energy  # normalize by window energy
        freq_bins, frames = X_stft.size(1), X_stft.size(2)
        X_stft = X_stft.reshape(B, C, freq_bins, frames)
            
        # 2) Estimate instantaneous frequency
        left  = X_stft[..., 1:2].flip(-1)       # (B, C, F, 1)
        right = X_stft[..., -2:-1].flip(-1)     # (B, C, F, 1)
        X_pad = torch.cat([left, X_stft, right], dim=-1)  # (B,C,F,L+2)

        dG = (X_pad[..., 2:] - X_pad[..., :-2]) / 2.0
        inst_freq = (dG / (1j * X_stft)).real
        bin_indices = (inst_freq * (self.context_window / (2*math.pi))).round().long()
        bin_indices = bin_indices.clamp(0, self.freq_bins - 1)
        
        # 3) Synchrosqueeze: reassign energy -> (B, C, freq_bins, frames)
        S = torch.zeros_like(X_stft)
        S = S.scatter_add(dim=2, index=bin_indices, src=X_stft)
        
        # 保持原来基于实部/虚部的扩展逻辑（可选）
        basis_cos = torch.einsum('bcfl,fl->bcfl', S.real, self.cos) # X_stft.real # (B, C, freq_bins, frames) real
        basis_sin = torch.einsum('bcfl,fl->bcfl', S.imag, self.sin) # X_stft.imag # imag
        S_proj = basis_cos + basis_sin
        
        # 5) Flatten channels & freq_bins 
        S_flat = self.flatten(S_proj)
        out = self.dropout(S_flat)
        # self.linear = nn.Linear(self.freq_bins * C, self.target_dim).to(out.device)
        out = self.linear(out)
        return out



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
            method="dft"
        ):
        super().__init__()
        self.decomposer_layer = SeriesDecomposer(kernel_size=kernel_size)
        self.method = method

        if method == "none":
            self.res_layer = DummyExtension(context_window * 2 // 3, target_window)
            self.trend_layer = DummyExtension(context_window * 2 // 3, target_window)
        elif method == "frsst":
            self.res_layer = FrSSTBiasProjection(
                context_window=context_window * 2 // 3,
                target_window=target_window
            )
            self.trend_layer = FrSSTBiasProjection(
                context_window=context_window * 2 // 3,
                target_window=target_window
            )
        elif method == "stft":
            self.res_layer = STFTBiasProjection(
                context_window=context_window * 2 // 3,
                target_window=target_window
            )
            self.trend_layer = STFTBiasProjection(
                context_window=context_window * 2 // 3,
                target_window=target_window
            )
        else:
            self.res_layer = DFTBasisExtension(
                context_window=context_window,
                target_window=target_window
            )
            self.trend_layer = DFTBasisExtension(
                context_window=context_window,
                target_window=target_window
            )

    def forward(self, x):
        res, trend = self.decomposer_layer(x) # # batch_size * num_features, num_steps, num_channels 
        res = res.permute(0, 2, 1) # batch_size * num_features, num_channels, num_steps
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
        self.res_fbm_layer = DFTBasisExtension(
            context_window=context_window,
            target_window=target_window
        )
        self.trend_fbm_layer = DFTBasisExtension(
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
