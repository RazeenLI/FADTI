import numpy as np
import torch
import torch.nn as nn

from .layers import BackboneBRITS

from nn.process_data import get_process_data

class BRITS(nn.Module):
    def __init__(
            self,
            data_name,
            num_features,
            # num_layers,
            # num_heads,
            num_channels,
            # num_diffusion_steps,
            num_steps,
            # dim_time_embedding,
            # dim_feature_embedding,
            # dim_diffusion_embedding,
            # is_unconditional,
            # schedule,
            # beta_start,
            # beta_end,
            # target_strategy,
            # method, # "dft" "stft" "frsst"
            # type_layer, # "attn" "conv" "atten+conv"
            device,
    ):
        super().__init__()
        self.device = device

        self.process_data = get_process_data(data_name)

        # self.dim_target = dim_target # target embedding dim
        # self.dim_time_embedding = dim_time_embedding # time embedding dim
        # self.dim_feature_embedding = dim_feature_embedding # feature embedding dim
        
        # self.is_unconditional = is_unconditional
        self.num_features = num_features
        self.num_channels = num_channels
        # self.num_diffusion_steps = num_diffusion_steps

        # Side Information (Conditional Information)
        # dim_side = dim_time_embedding + dim_feature_embedding
        # if self.is_unconditional:
        #     dim_input = 1
        # else:
        #     dim_input = 2
        #     dim_side += 1 # for conditional mask

        self.model = BackboneBRITS(
            n_steps=num_steps,
            n_features=num_features,
            rnn_hidden_size=num_channels,
        )
    
    def impute(
            self,
            observed_data,
            conditional_mask,
    ):
        
        batch_size, num_features, num_steps = observed_data.shape

        forward_delta = self.compute_deltas(conditional_mask, False)

        back_observed_data = observed_data.flip(dims=[2])
        back_conditional_mask = conditional_mask.flip(dims=[2])
        backward_deltas = forward_delta.flip(dims=[2])

        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1) 
        forward_delta = forward_delta.permute(0, 2, 1) 
        back_observed_data = back_observed_data.permute(0, 2, 1) 
        back_conditional_mask = back_conditional_mask.permute(0, 2, 1) 
        backward_deltas = backward_deltas.permute(0, 2, 1) 

        forward_data = {
            "data": observed_data,
            "mask": conditional_mask,
            "deltas": forward_delta,
        }

        backward_data = {
            "data": back_observed_data,
            "mask": back_conditional_mask,
            "deltas": backward_deltas,
        }

        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(forward_data, backward_data)

        # results = {
        #     "imputation": imputed_data.permute(0, 2, 1),
        #     "consistency_loss": consistency_loss,
        #     "reconstruction_loss": reconstruction_loss,
        #     "reconstruction": (f_reconstruction + b_reconstruction) / 2,
        #     "f_reconstruction": f_reconstruction,
        #     "b_reconstruction": b_reconstruction,
        # }

        return imputed_data.permute(0, 2, 1)
    
    def compute_deltas(self, masks: torch.Tensor, backward=False):
        """
        masks: [B, C, T] (1=observed, 0=missing)
        Returns: deltas of same shape
        """
        if backward:
            masks = torch.flip(masks, dims=[2])

        B, C, T = masks.shape
        deltas = torch.ones_like(masks)

        for t in range(1, T):
            deltas[:, :, t] = 1 + (1 - masks[:, :, t]) * deltas[:, :, t - 1]

        if backward:
            deltas = torch.flip(deltas, dims=[2])

        return deltas

        
    def calc_loss(
        self, 
        observed_data, 
        conditional_mask, 
        observed_mask,
    ):
        batch_size, num_features, num_steps = observed_data.shape

        forward_delta = self.compute_deltas(conditional_mask, False)

        back_observed_data = observed_data.flip(dims=[2])
        back_conditional_mask = conditional_mask.flip(dims=[2])
        backward_deltas = forward_delta.flip(dims=[2])

        observed_data = observed_data.permute(0, 2, 1) 
        conditional_mask = conditional_mask.permute(0, 2, 1) 
        forward_delta = forward_delta.permute(0, 2, 1) 
        back_observed_data = back_observed_data.permute(0, 2, 1) 
        back_conditional_mask = back_conditional_mask.permute(0, 2, 1) 
        backward_deltas = backward_deltas.permute(0, 2, 1) 

        forward_data = {
            "data": observed_data,
            "mask": conditional_mask,
            "deltas": forward_delta,
        }

        backward_data = {
            "data": back_observed_data,
            "mask": back_conditional_mask,
            "deltas": backward_deltas,
        }

        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(forward_data, backward_data)

        loss = consistency_loss + reconstruction_loss

        # target_mask = observed_mask - conditional_mask
        # residual = (noise - predicted) * target_mask
        # num_eval = target_mask.sum()
        # loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss
    
    def forward(
            self,
            inputs,
            num_sampling_times=1,
    ):
        results = {}
        if self.training:
            # Training
            # (observed_data, observed_mask, conditional_mask, observed_tp) = (
            #     inputs["X_ori"],
            #     inputs["observed_mask"],
            #     inputs["cond_mask"],
            #     inputs["observed_tp"],
            # )
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
            # side_info = self.get_side_info(observed_tp, conditional_mask)

            # training loss
            results["loss"] = self.calc_loss(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
                # side_info=side_info, 
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

            # side_info = self.get_side_info(observed_tp, conditional_mask)
            # validating loss
            results["loss"] = self.calc_loss(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
                # side_info=side_info, 
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
    