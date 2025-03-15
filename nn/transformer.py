import copy
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerEncoderLayer_QKV(nn.Module):
    def __init__(
            self, 
            dim_model, 
            num_heads, 
            dim_feedforward=2048, 
            dropout=0.1, 
            activation="relu"
        ):
        super(TransformerEncoderLayer_QKV, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)

        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim_model)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_QKV, self).__setstate__(state)

    def forward(
            self, 
            query, 
            key, 
            value, 
            attn_mask=None, 
            key_padding_mask=None
        ):
        value2 = self.self_attn(
            query, 
            key, 
            value, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )[0]
        value = value + self.dropout1(value2)
        value = self.norm1(value)
        value2 = self.linear2(self.dropout(self.activation(self.linear1(value))))
        value = value + self.dropout2(value2)
        value = self.norm2(value)
        return value

class TransformerEncoder_QKV(nn.Module):
    __constants__ = ['norm']

    def __init__(
            self, 
            encoder_layer, 
            num_layers, 
            norm=None
        ):
        super(TransformerEncoder_QKV, self).__init__()
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(encoder_layer) for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self, 
            query, 
            key, 
            value, 
            mask=None, 
            key_padding_mask=None
        ):
        for layer in self.layers:
            value = layer(
                query, 
                key, 
                value, 
                attn_mask=mask, 
                key_padding_mask=key_padding_mask
            )
        if self.norm is not None:
            value = self.norm(value)
        return value
