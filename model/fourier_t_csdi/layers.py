import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.transformer import TransformerEncoder_QKV, TransformerEncoderLayer_QKV
from nn.fourier import FourierBasisMapping

def get_transformer(num_heads=8, num_layers=1, num_channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        dim_model=num_channels,
        num_heads=num_heads,
        dim_feedforward=64,
        activation="gelu"
    )
    return TransformerEncoder_QKV(
        encoder_layer=encoder_layer,
        num_layers=num_layers
    )
    # encoder_layer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=num_heads, dim_feedforward=64, activation="gelu")
    # return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(
            self,
            num_steps,
            dim_embedding = 128,
            dim_projection = None, # dimensions of the final projection
    ):
        super().__init__()

        if dim_projection is None:
            dim_projection = dim_embedding

        # Precomputed Embedding Tables
        # Register Buffer
        # for tensor that related to the model state but do not need to be updateed during training
        self.register_buffer(
            "embedding", 
            self._build_embedding(num_steps, dim_embedding // 2), # The final table will be twice dim_embedding, so it needs to be divided by 2.
            persistent=False
        )

        self.projection_layer1 = nn.Linear(dim_embedding, dim_projection)
        self.projection_layer2 = nn.Linear(dim_projection, dim_projection)

    @staticmethod
    def _build_embedding(
            num_steps, 
            dim_embedding=64
    ):
        """
        Similar to the positional encoding in the Transformer, but here the embedding table is fixed for the number of diffusion steps.
        INPUT: 
        num_steps: number of diffusion steps, i.e. the number of rows in the generated embedding table
        dim_embedding: half dimension of the embedding vector
        OUTPUT: torch.Tensor
        """
        # (num_step, 1)(T,1) size tensor
        steps = torch.arange(num_steps).unsqueeze(1)
        # (1, dim_embedding)
        # 1. Values ​​in the interval [0,4]
        # 2. Exponential operation with base 10
        frequencies = 10.0 ** (torch.arange(dim_embedding) / (dim_embedding - 1) * 4.0).unsqueeze(0)
        # (num_step, dim_embedding)
        table = steps * frequencies
        # (num_step, dim_embedding*2)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


    def forward(
            self,
            diffusion_step: int
    ):
        """
        INPUT: 
        diffusion_step: index of the current diffusion step
        OUTPUT: diffusion embedding
        """
        # (dim_embedding)
        x = self.embedding[diffusion_step]
        x = self.projection_layer1(x)
        x = F.silu(x)
        x = self.projection_layer2(x)
        x = F.silu(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(
            self, 
            num_channels, 
            num_heads, 
            num_steps,
            num_layers=1,
            is_cross=False,
            init_cutoff_ratio=0.5,
            apply_ifft=True,
        ):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = get_transformer(
            num_heads=num_heads, 
            num_layers=num_layers, 
            num_channels=num_channels
        )
        self.fft_layer = FourierBasisMapping(
            context_window=num_steps,
            target_window=num_steps,
        )
        self.fusion_layer = nn.Linear(num_channels * 2, num_channels)
        self.norm = nn.LayerNorm(num_channels)  # 针对每个时间步内的 channel 归一化


    def forward(self, x, base_shape, itp_x=None):
        batch_size, num_channels, num_features, num_steps = base_shape
        # print(f"\n\n\n\n============= x.shape = {base_shape} =============\n\n\n\n")
        if num_steps == 1:
            return x
        # Time Domain
        v = x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps).permute(2, 0, 1)
        # Frequenct Domain
        v_fft = x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps).permute(0, 2, 1)
        v_fft = self.fft_layer(v_fft)
        v_fft = v_fft.permute(1, 0, 2)
        # v self imformation, q other information
        # combine
        # 简单平均融合（也可以采用加权或拼接后再映射 cat -> linear 的方式）
        # v = (v + x_fft) / 2
        # 加权
        # weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
        # v = weights[0] * v + weights[1] * x_fft
        # 拼接后再映射 cat -> linear
        v = torch.cat([v, v_fft], dim=-1)
        v = self.fusion_layer(v)
        # v = v_fft

        # if self.is_cross:
        #     q = itp_x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps).permute(2, 0, 1)
        #     x = self.time_layer(q, v, v).permute(1, 2, 0)
        # else:
        #     x = self.time_layer(v, v, v).permute(1, 2, 0)
        x = v.permute(1, 2, 0)

        x = x.reshape(batch_size, num_features, num_channels, num_steps).permute(0, 2, 1, 3).reshape(batch_size, num_channels, num_features * num_steps)
        return x
    

# class TemporalAttention(nn.Module):
#     def __init__(
#             self, 
#             num_channels, 
#             num_heads, 
#             num_layers=1,
#             is_cross=False,
#             init_cutoff_ratio=0.5,
#             apply_ifft=True,
#         ):
#         super().__init__()
#         self.is_cross = is_cross
#         self.time_layer = get_transformer(
#             num_heads=num_heads, 
#             num_layers=num_layers, 
#             num_channels=num_channels
#         )
#         self.fft_linear_layer = nn.Linear(num_channels * 2, num_channels)
#         self.fusion_layer = nn.Linear(num_channels * 2, num_channels)
#         # # 可学习的融合权重参数（初始值可以设为 0.5）
#         # self.alpha = nn.Parameter(torch.tensor(0.5))
#         # self.beta = nn.Parameter(torch.tensor(0.5))
#         self.norm = nn.LayerNorm(num_channels)  # 针对每个时间步内的 channel 归一化

#     def fft(self, x, base_shape):
#         batch_size, num_channels, num_features, num_steps = base_shape
#         # 将 x reshape 成 (batch_size, num_channels, num_features, num_steps)
#         x = x.reshape(batch_size, num_channels, num_features, num_steps)
#         # 沿着时间维度做 FFT，返回的是复数张量
#         x_fft = torch.fft.fft(x, dim=-1)


#         # 提取实部和虚部并拼接
#         x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)  # (batch_size, 2*num_channels, num_features, num_steps)
#         # 根据需要再 reshape 成 (batch_size, new_channels, num_features * num_steps)
#         x = x_fft.permute(0, 2, 1, 3).reshape(batch_size * num_features, 2*num_channels, num_steps)
#         x = x.permute(2, 0, 1) # (num_steps, batch_size*num_features, 2*num_channels)
#         # 通过线性层映射回 num_channels
#         x = self.fft_linear_layer(x)  # (num_steps, batch_size*num_features, num_channels)

#         # 应用 LayerNorm
#         x = self.norm(x)
#         return x

#     def forward(self, x, base_shape, itp_x=None):
#         batch_size, num_channels, num_features, num_steps = base_shape
#         if num_steps == 1:
#             return x
#         # Time Domain
#         v = x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps).permute(2, 0, 1)
#         # Frequenct Domain
#         x_fft = self.fft(x, base_shape)
#         # v self imformation, q other information
#         # combine
#         # 简单平均融合（也可以采用加权或拼接后再映射 cat -> linear 的方式）
#         # v = (v + x_fft) / 2
#         # 加权
#         # weights = torch.softmax(torch.stack([self.alpha, self.beta]), dim=0)
#         # v = weights[0] * v + weights[1] * x_fft
#         # 拼接后再映射 cat -> linear
#         v = torch.cat([v, x_fft], dim=-1)
#         v = self.fusion_layer(v)

#         if self.is_cross:
#             q = itp_x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps).permute(2, 0, 1)
#             x = self.time_layer(q, v, v).permute(1, 2, 0)
#         else:
#             x = self.time_layer(v, v, v).permute(1, 2, 0)

#         x = x.reshape(batch_size, num_features, num_channels, num_steps).permute(0, 2, 1, 3).reshape(batch_size, num_channels, num_features * num_steps)
#         return x

class FeatureAttention(nn.Module):
    def __init__(
            self, 
            num_channels, 
            num_heads,
            num_layers=1, 
            is_cross=False
        ):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = get_transformer(
            num_heads=num_heads, 
            num_layers=num_layers, 
            num_channels=num_channels
        )

    def forward(self, x, base_shape, itp_x=None):
        batch_size, num_channels, num_features, num_steps = base_shape
        if num_features == 1:
            return x
        v = x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features).permute(2, 0, 1)
        if self.is_cross:
            q = itp_x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features).permute(2, 0, 1)
            x = self.feature_layer(q, v, v).permute(1, 2, 0)
        else:
            x = self.feature_layer(v, v, v).permute(1, 2, 0)
        x = x.reshape(batch_size, num_steps, num_channels, num_features).permute(0, 2, 3, 1).reshape(batch_size, num_channels, num_features * num_steps)
        return x

class ResidualBlock(nn.Module):
    def __init__(
            self,
            dim_diffusion_embedding,
            dim_side,
            num_channels,
            num_heads,
            num_steps,
    ):
        super().__init__()
        self.diffusion_projection_layer = nn.Linear(dim_diffusion_embedding, num_channels)

        self.conditional_projection_layer = conv1d_with_init(dim_side, 2 * num_channels, 1)

        # Prepare for the subsequent gating mechanism
        self.middle_projection_layer = conv1d_with_init(num_channels, 2 * num_channels, 1)
        self.output_projection_layer = conv1d_with_init(num_channels, 2 * num_channels, 1)

        # change part 1
        self.time_layer = TemporalAttention(
            num_heads=num_heads,
            num_layers=1,
            num_channels=num_channels,
            num_steps=num_steps,
        )
        self.feature_layer = FeatureAttention(
            num_heads=num_heads,
            num_layers=1,
            num_channels=num_channels
        )

    # def forward_time(self, y, base_shape):
    #     batch_size, num_channels, num_features, num_steps = base_shape
    #     if num_steps == 1:
    #         return y
    #     y = y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps)
    #     y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
    #     y = y.reshape(batch_size, num_features, num_channels, num_steps).permute(0, 2, 1, 3).reshape(batch_size, num_channels, num_features * num_steps)
    #     return y
    
    # def forward_feature(self, y, base_shape):
    #     batch_size, num_channels, num_features, num_steps = base_shape
    #     if num_features == 1:
    #         return y
    #     y = y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features)
    #     y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
    #     y = y.reshape(batch_size, num_steps, num_channels, num_features).permute(0, 2, 3, 1).reshape(batch_size, num_channels, num_features * num_steps)
    #     return y


    def forward(
            self,
            x, # (bitch_size, num_channels, num_features, time_steps)(B,channels, K, L)
            conditional_info, # similar to x size, including additional info and mask info
            diffusion_embedding
    ):
        batch_size, num_channels, num_features, num_steps = x.shape
        base_shape = x.shape
        x = x.reshape(batch_size, num_channels, num_features * num_steps)

        diffusion_embedding = self.diffusion_projection_layer(diffusion_embedding).unsqueeze(-1) # (batch_size, num_channels, 1)

        y = x + diffusion_embedding
        y = self.time_layer(y, base_shape)
        y = self.feature_layer(y, base_shape) # (batch_size, num_channels, num_features * num_steps)
        y = self.middle_projection_layer(y) # (batch_size, 2*num_channels, num_features * num_steps)


        _, dim_side, _, _ = conditional_info.shape
        conditional_info = conditional_info.reshape(batch_size, dim_side, num_features * num_steps)
        conditional_info = self.conditional_projection_layer(conditional_info)  # (B,2*channel,K*L)
        y = y + conditional_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection_layer(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class DiffusionFTCSDI(nn.Module):
    def __init__(
            self,
            num_diffusion_steps,
            dim_diffusion_embedding,
            dim_input,
            dim_side,
            num_channels,
            num_heads,
            num_layers,
            num_steps,
    ):
        super().__init__()

        self.num_channels = num_channels

        self.defussion_embedding_layer = DiffusionEmbedding(
            num_steps=num_diffusion_steps,
            dim_embedding=dim_diffusion_embedding
        )

        self.input_projection_layer = conv1d_with_init(dim_input, num_channels, 1)
        self.output_projection_layer1 = conv1d_with_init(num_channels, num_channels, 1)
        self.output_projection_layer2 = conv1d_with_init(num_channels, 1, 1)
        # Zero Weight Initialization
        # Makes the output relatively stable in the initial state, which helps the stability in the early stage of training.
        nn.init.zeros_(self.output_projection_layer2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    dim_diffusion_embedding=dim_diffusion_embedding,
                    dim_side=dim_side,
                    num_channels=num_channels,
                    num_heads=num_heads,
                    num_steps=num_steps,
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(
            self,
            x,
            conditional_info,
            diffusion_step,
    ):
        (batch_size, dim_input, num_features, num_steps) = x.shape # n_samples, 2, n_features, n_steps

        x = x.reshape(batch_size, dim_input, num_features * num_steps)
        x = self.input_projection_layer(x)
        x = F.relu(x)
        x = x.reshape(batch_size, self.num_channels, num_features, num_steps)

        diffusion_embedding = self.defussion_embedding_layer(diffusion_step)

        skips = [] # skip connection list
        for layer in self.residual_layers:
            x, skip = layer(x, conditional_info, diffusion_embedding)
            skips.append(skip)
        
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(batch_size, self.num_channels, num_features * num_steps)
        x = self.output_projection_layer1(x)  # (n_samples, num_channel, n_features*n_steps)
        x = F.relu(x)
        x = self.output_projection_layer2(x)  # (n_samples, 1, n_features*n_steps)
        x = x.reshape(batch_size, num_features, num_steps)

        return x
        