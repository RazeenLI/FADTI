import torch.nn as nn
import torch
import numpy as np

from model.timemixer.layers import BackboneTimeMixer
from model.transformer.embeddings import DataEmbedding
from nn.process_data import get_process_data



class TimeMixer(nn.Module):
    def __init__(
        self,
        data_name,
        num_layers, # num_layers
        num_steps, # num_steps
        num_features, # num_features
        top_k,
        d_model,
        d_ffn,
        channel_independence,
        decomp_method,
        moving_avg,
        downsampling_layers,
        downsampling_window,
        dropout,
        device,
    ):
        super().__init__()

        self.device = device

        self.seq_len = num_steps
        self.n_layers = num_layers

        self.process_data = get_process_data(data_name)

        # self.enc_embedding = DataEmbedding(
        #     num_features,
        #     d_model,
        #     dropout=dropout,
        #     n_max_steps=num_steps,
        # )
        self.model = BackboneTimeMixer(
            task_name="imputation",
            n_steps=num_steps,
            n_features=num_features,
            n_pred_steps=0,
            n_pred_features=num_features,
            n_layers=num_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            channel_independence=channel_independence,
            decomp_method=decomp_method,
            top_k=top_k,
            moving_avg=moving_avg,
            downsampling_layers=downsampling_layers,
            downsampling_window=downsampling_window,
            downsampling_method="avg",
            use_future_temporal_feature=False,
        )
        # self.layer_norm = nn.LayerNorm(d_model)

        # # for the imputation task, the output dim is the same as input dim
        # self.projection = nn.Linear(d_model, num_features)

    def impute(
            self,
            observed_data,
            conditional_mask,
    ):
        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1) 

        reconstruction = self.model.imputation(observed_data, None)

        imputed_data = conditional_mask * observed_data + (1 - conditional_mask) * reconstruction
        # results = {
        #     "imputation": imputed_data,
        #     "reconstruction": reconstruction,
        # }
        # print("imputed_data.shape before permute", imputed_data.shape)
        # imputed_data.permute(0, 2, 1)
        # print("imputed_data.shape", imputed_data.shape)

        return imputed_data.permute(0, 2, 1)
    

    
    def calc_loss(
            self, 
            observed_data, 
            conditional_mask, 
            observed_mask,
    ):
        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)

        # embedding
        # input_data = self.enc_embedding(observed_data)  # [B,T,C]
        # # TimesNet processing
        # enc_out = self.model(input_data)

        # TimesMixer processing
        reconstruction = self.model.imputation(observed_data, None)

        target_mask = observed_mask - conditional_mask

        if self.training:  # if in the training mode (the training stage), return loss result from training_loss
            # `loss` is always the item for backward propagating to update the model
            loss = masked_mae_cal(reconstruction, observed_data, conditional_mask)
        else:  # if in the eval mode (the validation stage), return metric result from validation_metric
            # X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]
            loss = masked_mse_cal(reconstruction, observed_data, target_mask)

        return loss

    
    def forward(
            self,
            inputs,
            num_sampling_times=1,
    ):
        results = {}
        if self.training:
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

        # results = {}
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
    


def masked_mae_cal(inputs, target, mask):
    """calculate Mean Absolute Error"""
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)

def masked_mse_cal(inputs, target, mask):
    """calculate Mean Squared Error"""
    return torch.sum(((inputs - target) * mask) ** 2) / (torch.sum(mask) + 1e-9)
    