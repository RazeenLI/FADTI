"""
SAITS model for time-series imputation.

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from modeling.layers import *
# from modeling.utils import masked_mae_cal
from .layers import EncoderLayer, PositionalEncoding
from nn.process_data import get_process_data


class SAITS(nn.Module):
    def __init__(
        self,
        data_name,
        n_groups,
        n_group_inner_layers,
        dim_time,
        dim_feature,
        dim_model,
        dim_hidden,
        num_heads, # num_heads
        dim_k,
        dim_v,
        dropout,
        reconstruction_loss_weight,
        imputation_loss_weight,
        diagonal_attention_mask,
        device,
        param_sharing_strategy,
        input_with_mask,
        MIT,
        **kwargs
    ):
        super().__init__()
        self.process_data = get_process_data(data_name)
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = input_with_mask # kwargs["input_with_mask"]
        actual_d_feature = dim_feature * 2 if self.input_with_mask else dim_feature
        self.param_sharing_strategy = param_sharing_strategy # kwargs["param_sharing_strategy"]
        self.MIT = MIT # kwargs["MIT"]
        self.device = device # kwargs["device"]
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.imputation_loss_weight = imputation_loss_weight

        num_layers = n_group_inner_layers if self.param_sharing_strategy == "between_group" else n_groups

        self.layer_stack_for_first_block = nn.ModuleList(
            [
                EncoderLayer(
                    dim_time=dim_time, # d_time,
                    dim_feature=actual_d_feature, # actual_d_feature,
                    dim_model=dim_model, # d_model,
                    dim_hidden=dim_hidden, # d_inner,
                    num_heads=num_heads, # n_head,
                    dim_k=dim_k, # d_k,
                    dim_v=dim_v, # d_v,
                    dropout=dropout, # dropout,
                    attn_dropout=0,# 0,
                    diagonal_attention_mask=diagonal_attention_mask,
                    device=device
                ) for _ in range(num_layers)
            ]
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                EncoderLayer(
                    dim_time=dim_time, # d_time,
                    dim_feature=actual_d_feature, # actual_d_feature,
                    dim_model=dim_model, # d_model,
                    dim_hidden=dim_hidden, # d_inner,
                    num_heads=num_heads, # n_head,
                    dim_k=dim_k, # d_k,
                    dim_v=dim_v, # d_v,
                    dropout=dropout, # dropout,
                    attn_dropout=0,# 0,
                    diagonal_attention_mask=diagonal_attention_mask,
                    device=device
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(dim_model, n_position=dim_time)
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, dim_model)
        self.reduce_dim_z = nn.Linear(dim_model, dim_feature)
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_d_feature, dim_model)
        self.reduce_dim_beta = nn.Linear(dim_model, dim_feature)
        self.reduce_dim_gamma = nn.Linear(dim_feature, dim_feature)
        # for the 3rd block
        self.weight_combine = nn.Linear(dim_feature + dim_time, dim_feature)

    def impute(self, X, masks):
        # X, masks = inputs["X"], inputs["missing_mask"]
        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        # print(f"\n\n\n{input_X_for_first.shape}\n{masks.shape}\n\n\n")
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(
            self.position_enc(input_X_for_first)
        )  # namely term e in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        )
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(
            input_X_for_second
        )  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=2))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(
            self, 
            inputs, 
            num_sampling_times=1,
        ):
        results = {}
        # if self.training:
        # Training
        res = self.process_data(inputs, self.device)
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            # for_pattern_mask,
        ) = (
            res["observed_data"],
            res["observed_mask"],
            res["observed_tp"],
            res["gt_mask"],
            # res["for_pattern_mask"],
        )
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        # X, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0
        total_input = observed_data * gt_mask
        _, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(X=total_input, masks=observed_mask)

        reconstruction_loss += masked_mae_cal(X_tilde_1, total_input, observed_mask)
        reconstruction_loss += masked_mae_cal(X_tilde_2, total_input, observed_mask)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, total_input, observed_mask)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if (self.MIT or not self.training):
            # have to cal imputation loss in the val stage
            indicating_mask = observed_mask - gt_mask

            imputation_MAE = masked_mae_cal(
                X_tilde_3, observed_data, indicating_mask
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        results["loss"] = self.reconstruction_loss_weight * reconstruction_loss + self.imputation_loss_weight * imputation_MAE
        return results["loss"]

        # return {
        #     "imputed_data": imputed_data,
        #     # "reconstruction_loss": reconstruction_loss,
        #     "imputation_loss": imputation_MAE,
        #     # "reconstruction_MAE": final_reconstruction_MAE,
        #     "imputation_MAE": imputation_MAE,
        # }
    
    def evaluate(
            self,
            inputs,
            num_sampling_times=1,
    ):
        res = self.process_data(inputs, self.device)
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
        ) = (
            res["observed_data"],
            res["observed_mask"],
            res["observed_tp"],
            res["gt_mask"],
            res["cut_length"],
        )
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        # reconstruction_loss = 0
        total_input = observed_data * gt_mask
        imputed_data, _ = self.impute(X=total_input, masks=observed_mask)

        target_mask = observed_mask - gt_mask

        for i in range(len(cut_length)):  # to avoid double evaluation
            target_mask[i, ..., 0 : cut_length[i].item()] = 0

        # for process
        imputed_data = imputed_data.unsqueeze(1)

        return imputed_data, observed_data, target_mask, observed_mask, observed_tp


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
    