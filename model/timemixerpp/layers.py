import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.revin.layers import RevIN
from model.transformer.attention import ScaledDotProductAttention, MultiHeadAttention
from model.transformer.embeddings import DataEmbedding
from model.inception.layers import InceptionBlockV1, InceptionTransBlockV1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if len(frequency_list) < k:
        k = len(frequency_list)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    index = np.where(period > 0)
    top_list = top_list[index]
    period = period[period > 0]
    return period, abs(xf).mean(-1)[:, top_list], top_list


class RowAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super().__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
        K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
        V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        row_attn = torch.bmm(Q, K)
        row_attn = self.softmax(row_attn)
        out = torch.bmm(V, row_attn.permute(0, 2, 1))
        out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
        out = self.gamma * out + x
        return out


class ColAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim):
        super().__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # size = (b*w,h,c2)
        K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c2,h)
        V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)  # size = (b*w,c1,h)

        col_attn = torch.bmm(Q, K)
        col_attn = self.softmax(col_attn)
        out = torch.bmm(V, col_attn.permute(0, 2, 1))
        out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
        out = self.gamma * out + x
        return out


class MultiScaleSeasonCross(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()
        self.cross_conv_season = nn.Sequential(
            InceptionBlockV1(
                d_model,
                d_ff,
                num_kernels=num_kernels,
                stride=(downsampling_window, 1),
            ),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, season_list):
        B, N, _, _ = season_list[0].size()
        # cross high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 3, 1).reshape(B, -1, N)]
        for i in range(len(season_list) - 1):
            out_low_res = self.cross_conv_season(out_high)
            # out_low_res = match_shape(out_low_res, out_low)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        return out_season_list


class MultiScaleTrendCross(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()

        self.cross_trans_conv_season = InceptionTransBlockV1(
            d_model,
            d_ff,
            num_kernels=num_kernels,
            stride=(downsampling_window, 1),
        )
        self.cross_trans_conv_season_restore = nn.Sequential(
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, trend_list):
        B, N, _, _ = trend_list[0].size()
        # cross low->high
        trend_list.reverse()
        out_low = trend_list[0]
        out_high = trend_list[1]
        out_trend_list = [out_low.permute(0, 2, 3, 1).reshape(B, -1, N)]

        for i in range(len(trend_list) - 1):
            out_high_res = self.cross_trans_conv_season(out_low, output_size=out_high.size())
            out_high_res = self.cross_trans_conv_season_restore(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list) - 1:
                out_high = trend_list[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        out_trend_list.reverse()
        return out_trend_list


class MixerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        top_k,
        d_model,
        d_ff,
        num_kernels,
        downsampling_window,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.downsampling_window = downsampling_window
        self.k = top_k
        self.layer_norm = nn.LayerNorm(d_model)
        self.row_attn_2d_trend = RowAttention(d_model, d_ff)
        self.col_attn_2d_season = ColAttention(d_model, d_ff)
        self.multi_scale_season_conv = MultiScaleSeasonCross(d_model, d_ff, num_kernels, downsampling_window)
        self.multi_scale_trend_conv = MultiScaleTrendCross(d_model, d_ff, num_kernels, downsampling_window)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        period_list, period_weight, top_list = FFT_for_Period(x_list[-1], self.k)

        res_list = []
        for i in range(len(period_list)):
            period = period_list[i]
            season_list = []
            trend_list = []
            for x in x_list:
                out = self.time_imaging(x, period)
                season, trend = self.dual_axis_attn(out)
                season_list.append(season)
                trend_list.append(trend)

            out_list = self.multi_scale_mixing(season_list, trend_list, length_list)
            res_list.append(out_list)

        res_list_new = []
        for i in range(len(x_list)):
            list = []
            for j in range(len(period_list)):
                list.append(res_list[j][i])
            res = torch.stack(list, dim=-1)
            res_list_new.append(res)

        res_list_agg = []
        for x, res in zip(x_list, res_list_new):
            res = self.multi_reso_mixing(period_weight, x, res)
            res = self.layer_norm(res)
            res_list_agg.append(res)
        return res_list_agg

    def time_imaging(self, x, period):
        B, T, N = x.size()
        out, length = self.__conv_padding(x, period)
        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
        return out

    def dual_axis_attn(self, out):
        trend = self.row_attn_2d_trend(out)
        season = self.col_attn_2d_season(out)
        return season, trend

    def multi_scale_mixing(self, season_list, trend_list, length_list):
        out_season_list = self.multi_scale_season_conv(season_list)
        out_trend_list = self.multi_scale_trend_conv(trend_list)
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

    @staticmethod
    def __conv_padding(x, period, downsampling_window=1):
        B, T, N = x.size()

        if T % (period * downsampling_window) != 0:
            length = ((T // (period * downsampling_window)) + 1) * period * downsampling_window
            padding = torch.zeros([B, (length - T), N]).to(x.device)
            out = torch.cat([x, padding], dim=1)
        else:
            length = T
            out = x
        return out, length

    @staticmethod
    def multi_reso_mixing(period_weight, x, res):
        B, T, N = x.size()
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class BackboneTimeMixerPP(nn.Module):
    def __init__(
        self,
        task_name: str,
        n_steps: int,
        n_features: int,
        n_pred_steps: int,
        n_pred_features: int,
        n_layers: int,
        d_model: int,
        d_ffn: int,
        n_heads: int,
        dropout: float,
        top_k: int,
        n_kernels: int,
        channel_mixing: bool,
        channel_independence: bool,
        downsampling_layers: int,
        downsampling_window: int,
        downsampling_method: str,
        use_future_temporal_feature: bool,
        use_norm: bool = False,
        embed="fixed",
        freq="h",
        n_classes=None,
    ):
        super().__init__()
        self.task_name = task_name
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_pred_steps = n_pred_steps
        self.n_pred_features = n_pred_features
        self.n_layers = n_layers
        self.channel_mixing = channel_mixing
        self.channel_independence = channel_independence
        self.downsampling_window = downsampling_window
        self.downsampling_layers = downsampling_layers
        self.downsampling_method = downsampling_method
        self.use_norm = use_norm
        self.use_future_temporal_feature = use_future_temporal_feature

        assert downsampling_method in ["max", "avg", "conv"], "downsampling_method must be in ['max', 'avg', 'conv']"

        if self.channel_independence:
            self.enc_embedding = DataEmbedding(1, d_model, embed, freq, dropout, with_pos=False)
        else:
            self.enc_embedding = DataEmbedding(n_features, d_model, embed, freq, dropout, with_pos=False)

        if self.use_norm:
            self.revin_layers = torch.nn.ModuleList([RevIN(n_features) for _ in range(downsampling_layers + 1)])

        self.encoder_model = nn.ModuleList(
            [
                MixerBlock(
                    n_steps,
                    n_pred_steps,
                    top_k,
                    d_model,
                    d_ffn,
                    n_kernels,
                    downsampling_window,
                )
                for _ in range(n_layers)
            ]
        )

        if self.channel_mixing:
            assert n_steps >= downsampling_window**downsampling_layers
            d_time_model = n_steps // (downsampling_window**downsampling_layers)
            d_kv = d_time_model // n_heads
            if d_kv == 0:
                d_kv = 1
            full_attn_operator = ScaledDotProductAttention(d_kv**0.5, dropout)
            self.channel_mixing_attention = MultiHeadAttention(
                full_attn_operator,
                d_time_model,
                n_heads,
                d_kv,
                d_kv,
            )

        if self.downsampling_method == "max":
            self.down_pool = torch.nn.MaxPool1d(self.downsampling_window, return_indices=False)
        elif self.downsampling_method == "avg":
            self.down_pool = torch.nn.AvgPool1d(self.downsampling_window)
        elif self.downsampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2

            if self.channel_independence:
                in_channels = 1
            else:
                in_channels = n_features

            self.down_pool = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=padding,
                stride=self.downsampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            raise ValueError("Downsampling method is error,only supporting the max, avg, conv1D")

        if task_name == "long_term_forecast" or task_name == "short_term_forecast":
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        n_steps // (downsampling_window**i),
                        n_pred_steps,
                    )
                    for i in range(downsampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
        elif task_name == "imputation" or task_name == "anomaly_detection":
            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, n_pred_features, bias=True)
        elif task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * n_steps, n_classes)
        else:
            raise NotImplementedError("Task not supported")

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.downsampling_layers):
            if self.downsampling_method == "conv" and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()

            x_enc_sampling = self.down_pool(x_enc_ori)

            if self.downsampling_method == "conv":
                x_enc_sampling_list.append(
                    x_enc_sampling.reshape(B, N, T // (self.downsampling_window ** (i + 1))).permute(0, 2, 1)
                )
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, :: self.downsampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, :: self.downsampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()

                x = self.revin_layers[i](x, x_mark, mode="norm") if self.use_norm else x
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.revin_layers[i](x, mode="norm") if self.use_norm else x
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.channel_mixing and self.channel_independence == 1:
            _, T, D = x_list[-1].size()

            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        if x_mark_enc is not None:
            for x, x_mark in zip(x_list, x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for x in x_list:
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out_list = []
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.n_pred_features, -1).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)

        dec_out = self.revin_layers[0](dec_out, mode="denorm") if self.use_norm else dec_out
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc
        enc_out_list = []

        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layer):
            enc_out_list = self.encoder_model[i](enc_out_list)

        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.revin_layers[i](x, "norm") if self.use_norm else x
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        if self.channel_mixing and self.channel_independence:
            _, T, D = x_list[-1].size()
            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.revin_layers[0](dec_out, "denorm") if self.use_norm else dec_out
        return dec_out

    def imputation(self, x_enc, x_mark_enc):

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        if self.channel_mixing and self.channel_independence:
            _, T, D = x_list[-1].size()
            coarse_scale_enc_out = x_list[-1].reshape(B, N, T * D)
            coarse_scale_enc_out, _ = self.channel_mixing_attention(
                coarse_scale_enc_out, coarse_scale_enc_out, coarse_scale_enc_out, None
            )
            x_list[-1] = coarse_scale_enc_out.reshape(B * N, T, D) + x_list[-1]

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.n_layers):
            enc_out_list = self.encoder_model[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.n_pred_features, -1).permute(0, 2, 1).contiguous()

        return dec_out

def match_shape(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Interpolate tensor `a` to match the last two dimensions of tensor `b` if needed.
    Assumes input shape: [B, C, T, D]
    """
    if a.shape[2:] != b.shape[2:]:
        a = F.interpolate(a, size=b.shape[2:], mode='nearest')
    return a
