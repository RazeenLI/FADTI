import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from modeling.layers import *
# from modeling.utils import masked_mae_cal
# from .layers import EncoderLayer, PositionalEncoding
from nn.process_data import get_process_data
from .layers import BackboneSAITS
from ..timesnet.model import masked_mae_cal, masked_mse_cal


class SAITS(nn.Module):
    def __init__(
        self,
        data_name,
        n_groups, # num_layers
        n_group_inner_layers,
        dim_time,
        dim_feature,
        dim_model,
        dim_hidden, # d_ffn
        num_heads, # num_heads
        dim_k,
        dim_v,
        dropout,
        reconstruction_loss_weight, # ORT_weight
        imputation_loss_weight, # MIT_weight
        diagonal_attention_mask,
        device,
        param_sharing_strategy,
        input_with_mask,
        MIT,
        **kwargs
    ):
        
        super().__init__()
        self.process_data = get_process_data(data_name)

        self.n_layers = n_groups
        self.n_steps = dim_time
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = reconstruction_loss_weight
        self.MIT_weight = imputation_loss_weight
        self.MIT = MIT
        # self.training_loss = training_loss

        self.encoder = BackboneSAITS(
            n_steps=dim_time,
            n_features=dim_feature,
            n_layers=n_groups,
            d_model=dim_model,
            n_heads=num_heads,
            d_k=dim_k,
            d_v=dim_v,
            d_ffn=dim_hidden,
            dropout=dropout,
            attn_dropout=0,
        )
        self.device = device # kwargs["device"]

    def impute(
            self,
            observed_data,
            conditional_mask,
    ):
        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1)
        # observed_mask = observed_mask.permute(0, 2, 1)

        if self.diagonal_attention_mask:
            mask_time = (1 - torch.eye(self.n_steps)).to(self.device).unsqueeze(0)
        else:
            mask_time = None

        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(observed_data, conditional_mask, mask_time)

        # replace the observed part with values from X
        imputed_data = conditional_mask * observed_data + (1 - conditional_mask) * X_tilde_3

        imputed_data = imputed_data.permute(0, 2, 1)
        return imputed_data

    def calc_loss(
            self, 
            observed_data, 
            conditional_mask, 
            observed_mask,
    ):
        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        if self.diagonal_attention_mask:
            mask_time = (1 - torch.eye(self.n_steps)).to(self.device).unsqueeze(0)
        else:
            mask_time = None
        
        # print("mask_time.shape", mask_time.shape)
        input_data = observed_data * conditional_mask

        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(input_data, conditional_mask, mask_time)

        reconstruction_loss = 0

        reconstruction_loss += masked_mae_cal(X_tilde_1, input_data, conditional_mask)
        reconstruction_loss += masked_mae_cal(X_tilde_2, input_data, conditional_mask)
        reconstruction_loss += masked_mae_cal(X_tilde_3, input_data, conditional_mask)
        # reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        if not self.training:
            # have to cal imputation loss in the val stage
            target_mask = observed_mask - conditional_mask

            imputation_MAE = masked_mae_cal(X_tilde_3, observed_data, target_mask)
        else:
            imputation_MAE = torch.tensor(0.0)

        loss = self.ORT_weight * reconstruction_loss + self.MIT_weight * imputation_MAE
        return loss



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

        results = {}
        if self.training:
            # Training
            # conditional_mask = self.get_randmask(observed_mask)
            conditional_mask = gt_mask

            # training loss
            results["loss"] = self.calc_loss(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
            )
            
        elif not self.training:
            # Validating
            conditional_mask = gt_mask

            # validating loss
            results["loss"] = self.calc_loss(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
            )
        return results["loss"]
    
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
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            samples = self.impute(observed_data, cond_mask)
            samples = samples.unsqueeze(1)  # (B, 1, F, T) batch_size, num_sampling_times, num_features, num_steps

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask