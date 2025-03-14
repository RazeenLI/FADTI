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
            target_window
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
        x = self.linear_layer(x)
        
        return x
    

class FBMMLP(nn.Module):
    def __init__(
            self,
            context_window,
            target_window
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
        self.linear_layer = nn.Sequential(
            nn.Linear(linear_input,720*2),      
            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Linear(720*2, 720*2),
            nn.Dropout(p=0.15),
            nn.ReLU(),
            nn.Linear(720*2, target_window)
        ) 

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
        x = self.linear_layer(x)
        
        return x
        

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
