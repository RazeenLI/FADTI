import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = TransformerEncoderLayer_QKV(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return TransformerEncoder_QKV(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(self, query, key, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(query, key, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, key, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(query, key, output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class AdaptiveGCN(nn.Module):
    def __init__(
            self, 
            num_channels, 
            order=2, 
            include_self=True, 
            # is_adp=True, 
        ):
        super().__init__()
        self.order = order
        self.include_self = include_self
        
        self.support_len = 2
        # self.is_adp = is_adp
        # if is_adp:
        #     self.support_len += 1

        c_in = num_channels
        c_out = num_channels

        c_in = (order * self.support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(
            self, 
            x, 
            base_shape, 
            # support_adp
        ):
        # B, channel, K, L = base_shape
        batch_size, num_channels, num_features, num_steps = base_shape
        if num_features == 1:
            return x
        # if self.is_adp:
        #     nodevec1 = support_adp[-1][0]
        #     nodevec2 = support_adp[-1][1]
        #     support = support_adp[:-1]
        # else:
        #     support = support_adp
        x = x.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features)

        if x.dim() < 4:
            is_squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            is_squeeze = False

        out = [x] if self.include_self else []
        # if (type(support) is not list):
        #     support = [support]
        # if self.is_adp:
        #     adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        #     support = support + [adp]
        # for a in support:
        #     x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
        #     out.append(x1)
        #     for k in range(2, self.order + 1):
        #         x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
        #         out.append(x2)
        #         x1 = x2
        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if is_squeeze:
            out = out.squeeze(-1)
        out = out.reshape(batch_size, num_steps, num_channels, num_features).permute(0, 2, 3, 1).reshape(batch_size, num_channels, num_features * num_steps)
        return out
    

class TemporalLearning(nn.Module):
    def __init__(self, num_channels, num_heads, is_cross=True):
        super().__init__()
        self.is_cross = is_cross
        self.time_layer = get_torch_trans(
            heads=num_heads, 
            layers=1, 
            channels=num_channels
        )

    def forward(self, y, base_shape, itp_y=None):
        batch_size, num_channels, num_features, num_steps = base_shape
        if num_steps == 1:
            return y
        y = y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps)
        v = y.permute(2, 0, 1)
        if self.is_cross:
            itp_y = itp_y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 2, 1, 3).reshape(batch_size * num_features, num_channels, num_steps)
            q = itp_y.permute(2, 0, 1)
            y = self.time_layer(q, q, v).permute(1, 2, 0)
        else:
            y = self.time_layer(v, v, v).permute(1, 2, 0)
        y = y.reshape(batch_size, num_features, num_channels, num_steps).permute(0, 2, 1, 3).reshape(batch_size, num_channels, num_features * num_steps)
        return y



class SpatialLearning(nn.Module):
    def __init__(
            self, 
            num_channels, 
            num_heads, 
            dim_target, 
            order, 
            include_self, 
            # is_adp, 
            proj_t, 
            is_cross
    ):
        super().__init__()
        self.is_cross = is_cross
        self.feature_layer = SpaDependLearning(
            num_channels=num_channels, 
            num_heads=num_heads, 
            order=order, 
            dim_target=dim_target,
            include_self=include_self,  
            # is_adp=is_adp, 
            proj_t=proj_t, is_cross=is_cross
        )

    def forward(self, y, base_shape, support, itp_y=None):
        batch_size, num_channels, num_features, num_steps = base_shape
        if num_features == 1:
            return y
        y = self.feature_layer(y, base_shape, support, itp_y)
        return y


class SpaDependLearning(nn.Module):
    def __init__(
            self, 
            num_channels, 
            num_heads, 
            dim_target, 
            order, 
            include_self, 
            # is_adp, 
            proj_t, 
            is_cross=True
        ):
        super().__init__()
        self.is_cross = is_cross

        self.gcn = AdaptiveGCN(
            num_channels=num_channels, 
            order=order, 
            include_self=include_self, 
            # is_adp=is_adp, 
        )

        self.feature_layer = SpatialAttention(
            num_channels=num_channels, 
            dim_target=dim_target, 
            k=proj_t, 
            num_heads=num_heads
        )
        self.cond_proj = conv1d_with_init(2 * num_channels, num_channels, 1)
        self.norm1_local = nn.GroupNorm(4, num_channels)
        self.norm1_attn = nn.GroupNorm(4, num_channels)
        self.ff_linear1 = nn.Linear(num_channels, num_channels * 2)
        self.ff_linear2 = nn.Linear(num_channels * 2, num_channels)
        self.norm2 = nn.GroupNorm(4, num_channels)

    def forward(
            self, 
            y, 
            base_shape, 
            # support, 
            itp_y=None
        ):
        # B, channel, K, L = base_shape
        batch_size, num_channels, num_features, num_steps = base_shape
        
        y_in1 = y

        # GCN
        y_gcn = self.gcn(y, base_shape)  # self.gcn(y, base_shape, support)       # [batch_size, num_channels, num_features*num_steps]
        y_gcn = y_in1 + y_gcn
        y_gcn = self.norm1_local(y_gcn)

        # feature layer
        y_attn = y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features)
        if self.is_cross:
            itp_y_attn = itp_y.reshape(batch_size, num_channels, num_features, num_steps).permute(0, 3, 1, 2).reshape(batch_size * num_steps, num_channels, num_features)
            y_attn = self.feature_layer(y_attn.permute(0, 2, 1), itp_y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y_attn = self.feature_layer(y_attn.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn = y_attn.reshape(batch_size, num_steps, num_channels, num_features).permute(0, 2, 3, 1).reshape(batch_size, num_channels, num_features * num_steps)
        
        y_attn = y_in1 + y_attn
        y_attn = self.norm1_attn(y_attn)

        y_in2 = y_gcn + y_attn
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


class GuidanceConstruct(nn.Module):
    def __init__(self, channels, nheads, target_dim, order, include_self, device, is_adp, adj_file, proj_t):
        super().__init__()
        self.GCN = AdaptiveGCN(channels, order=order, include_self=include_self, device=device, is_adp=is_adp, adj_file=adj_file)
        self.attn_s = SpatialAttention(dim=channels, seq_len=target_dim, k=proj_t, heads=nheads)
        self.attn_t = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.norm1_local = nn.GroupNorm(4, channels)
        self.norm1_attn_s = nn.GroupNorm(4, channels)
        self.norm1_attn_t = nn.GroupNorm(4, channels)
        self.ff_linear1 = nn.Linear(channels, channels * 2)
        self.ff_linear2 = nn.Linear(channels * 2, channels)
        self.norm2 = nn.GroupNorm(4, channels)


    def forward(self, y, base_shape, support):
        B, channel, K, L = base_shape
        y_in1 = y

        y_local = self.GCN(y, base_shape, support)       # [B, C, K*L]
        y_local = y_in1 + y_local
        y_local = self.norm1_local(y_local)

        y_attn_s1 = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y_attn_s = self.attn_s(y_attn_s1.permute(0, 2, 1)).permute(0, 2, 1)
        y_attn_s = y_attn_s.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        y_attn_s = y_in1 + y_attn_s
        y_attn_s = self.norm1_attn_s(y_attn_s)

        y_attn_t1 = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        v = y_attn_t1.permute(2, 0, 1)
        y_attn_t = self.attn_t(v, v, v).permute(1, 2, 0)
        y_attn_t = y_attn_t.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        y_attn_t = y_in1 + y_attn_t
        y_attn_t = self.norm1_attn_t(y_attn_t)

        y_in2 = y_local + y_attn_s + y_attn_t
        y = F.relu(self.ff_linear1(y_in2.permute(0, 2, 1)))
        y = self.ff_linear2(y).permute(0, 2, 1)
        y = y + y_in2

        y = self.norm2(y)
        return y


def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class SpatialAttention(nn.Module):
    def __init__(
            self, 
            num_channels, 
            dim_target, 
            k=256, 
            num_heads=8, 
            dim_head=None, 
            one_kv_head=False, 
            share_kv=False, 
            dropout=0.
        ):

        # self.attn = SpatialAttention(
        #     dim=num_channels, 
        #     seq_len=dim_target, 
        #     k=proj_t, 
        #     heads=num_heads
        # )
        
        super().__init__()
        assert (num_channels % num_heads) == 0, 'dimension must be divisible by the number of heads'

        self.dim_target = dim_target
        self.k = k
        self.num_heads = num_heads

        dim_head = default(dim_head, num_channels // num_heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(num_channels, dim_head * num_heads, bias=False)
        kv_dim = dim_head if one_kv_head else (dim_head * num_heads)
        self.to_k = nn.Linear(num_channels, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(dim_target, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(num_channels, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(dim_target, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * num_heads, num_channels)

    def forward(self, x, itp_x=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.num_heads, self.k

        v_len = n if itp_x is None else itp_x.shape[1]
        assert v_len == self.seq_len, f'the sequence length of the values must be {self.seq_len} - {v_len} given'

        q_input = x if itp_x is None else itp_x
        queries = self.to_q(q_input)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        k_input = x if itp_x is None else itp_x
        v_input = x

        keys = self.to_k(k_input)
        values = self.to_v(v_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values
        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class NoiseProject(nn.Module):

    def __init__(
            self, 
            dim_side, 
            num_channels, 
            dim_diffusion_embedding, 
            num_heads, 
            target_dim, 
            proj_t, 
            order=2, 
            include_self=True,
            # is_adp=False,  
            is_cross_t=False, 
            is_cross_s=True
        ):
        super().__init__()
        self.diffusion_projection_layer = nn.Linear(dim_diffusion_embedding, num_channels)

        self.conditional_projection_layer = conv1d_with_init(dim_side, 2 * num_channels, 1)

        # Prepare for the subsequent gating mechanism
        self.middle_projection_layer = conv1d_with_init(num_channels, 2 * num_channels, 1)
        self.output_projection_layer = conv1d_with_init(num_channels, 2 * num_channels, 1)


        self.forward_time = TemporalLearning(
            num_channels=num_channels, 
            num_heads=num_heads, 
            is_cross=is_cross_t
        )
        self.forward_feature = SpatialLearning(
            num_channels=num_channels, 
            num_heads=num_heads, 
            dim_target=target_dim,
            order=order, 
            include_self=include_self,
            # is_adp=is_adp,
            proj_t=proj_t, 
            is_cross=is_cross_s
        )

    def forward(
            self, 
            x, 
            conditional_info, 
            diffusion_embedding, 
            itp_info, 
            # support, # support_adp
        ):
    #     def forward(
    #         self,
    #         x, # (bitch_size, num_channels, num_features, time_steps)(B,channels, K, L)
    #         conditional_info, # similar to x size, including additional info and mask info
    #         diffusion_embedding
    # ):
        # B, channel, K, L = x.shape
        batch_size, num_channels, num_features, num_steps = x.shape
        base_shape = x.shape
        x = x.reshape(batch_size, num_channels, num_features * num_steps)

        diffusion_embedding = self.diffusion_projection_layer(diffusion_embedding).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_embedding

        y = self.forward_time(y, base_shape, itp_info)
        y = self.forward_feature(y, base_shape, itp_info)  # (B,channel,K*L) # self.forward_feature(y, base_shape, support, itp_info)  # (B,channel,K*L)
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


class DiffusionPriSTI(nn.Module):
    def __init__(
            self, 
            device,
            num_channels,
            num_heads,
            num_diffusion_steps,
            num_layers,
            num_features,
            dim_input,
            dim_side, 
            dim_diffusion_embedding,
            proj_t,
            is_cross_t,
            is_cross_s,
            # is_adp=False,
            # adj_file,
            # is_itp=False
        ):
        super().__init__()

        self.num_channels = num_channels
        # self.is_itp = is_itp
        self.itp_channels = None

        # if self.is_itp:
        #     self.itp_channels = self.num_channels
        #     self.itp_projection_layer1 = conv1d_with_init(dim_input-1, self.itp_channels, 1)

        #     self.itp_modeling_layer = GuidanceConstruct(
        #         channels=self.itp_channels, 
        #         nheads=num_heads, 
        #         target_dim=target_dim,
        #         order=2, 
        #         include_self=True, 
        #         device=device,
        #         is_adp=config["is_adp"],
        #         adj_file=config["adj_file"], 
        #         proj_t=config["proj_t"]
        #     )
        #     self.cond_projection_layer = conv1d_with_init(dim_side, self.itp_channels, 1)
        #     self.itp_projection_layer2 = conv1d_with_init(self.itp_channels, 1, 1)

        self.defussion_embedding_layer = DiffusionEmbedding(
            num_steps=num_diffusion_steps,
            embedding_dim=dim_diffusion_embedding,
        )

        # if adj_file == 'AQI36':
        #     self.adj = get_adj_AQI36()
        # elif adj_file == 'metr-la':
        #     self.adj = get_similarity_metrla(thr=0.1)
        # elif adj_file == 'pems-bay':
        #     self.adj = get_similarity_pemsbay(thr=0.1)

        self.device = device

        # self.support = compute_support_gwn(self.adj, device=device)
        # adj_mx = [asym_adj(self.adj), asym_adj(np.transpose(self.adj))]
        # self.support = [torch.tensor(i).to(device) for i in adj_mx]
        # self.is_adp = is_adp

        # if self.is_adp:
        #     node_num = self.adj.shape[0]
        #     self.nodevec1 = nn.Parameter(torch.randn(node_num, 10).to(self.device), requires_grad=True).to(self.device)
        #     self.nodevec2 = nn.Parameter(torch.randn(10, node_num).to(self.device), requires_grad=True).to(self.device)
        #     self.support.append([self.nodevec1, self.nodevec2])

        self.input_projection_layer = conv1d_with_init(dim_input, num_channels, 1)
        self.output_projection_layer1 = conv1d_with_init(num_channels, num_channels, 1)
        self.output_projection_layer2 = conv1d_with_init(num_channels, 1, 1)
        nn.init.zeros_(self.output_projection_layer2.weight)

        self.residual_layers = nn.ModuleList(
            [
                NoiseProject(
                    dim_side=dim_side,
                    num_channels=num_channels,
                    dim_diffusion_embedding=dim_diffusion_embedding,
                    num_heads=num_heads,
                    target_dim=num_features,
                    proj_t=proj_t,
                    # is_adp=is_adp,
                    is_cross_t=is_cross_t,
                    is_cross_s=is_cross_s,
                    # proj_t=config["proj_t"],
                    # is_adp=config["is_adp"],
                    # is_cross_t=config["is_cross_t"],
                    # is_cross_s=config["is_cross_s"],
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
            self, 
            x, 
            conditional_info, 
            diffusion_step, 
            itp_x, 
        ):
        # if self.is_itp:
        #     x = torch.cat([x, itp_x], dim=1)
        batch_size, dim_input, num_features, num_steps = x.shape
        # B, inputdim, K, L = x.shape

        x = x.reshape(batch_size, dim_input, num_features * num_steps)
        x = self.input_projection_layer(x)
        x = F.relu(x)
        x = x.reshape(batch_size, self.num_channels, num_features, num_steps)

        # if self.is_itp:
        #     itp_x = itp_x.reshape(B, inputdim-1, K * L)
        #     itp_x = self.itp_projection(itp_x)
        #     itp_cond_info = side_info.reshape(B, -1, K * L)
        #     itp_cond_info = self.cond_projection(itp_cond_info)
        #     itp_x = itp_x + itp_cond_info
        #     itp_x = self.itp_modeling(itp_x, [B, self.itp_channels, K, L], self.support)
        #     itp_x = F.relu(itp_x)
        #     itp_x = itp_x.reshape(B, self.itp_channels, K, L)

        diffusion_embedding = self.defussion_embedding_layer(diffusion_step)

        skips = [] # skip connection list
        for layer in self.residual_layers:
            # x, skip = layer(x, conditional_info, diffusion_embedding, itp_x, self.support)
            x, skip = layer(x, conditional_info, diffusion_embedding, itp_x)
            skips.append(skip)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(batch_size, self.num_channels, num_features * num_steps)
        x = self.output_projection_layer1(x)  # (n_samples, num_channel, n_features*n_steps)
        x = F.relu(x)
        x = self.output_projection_layer2(x)  # (n_samples, 1, n_features*n_steps)
        x = x.reshape(batch_size, num_features, num_steps)
        return x