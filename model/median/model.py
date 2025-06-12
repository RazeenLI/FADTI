import numpy as np
import torch
import torch.nn as nn

from nn.process_data import get_process_data

class Median(nn.Module):
    def __init__(
            self,
            data_name,
            device,
    ):
        super().__init__()
        self.device = device

        self.process_data = get_process_data(data_name)

    def forward(
            self,
            inputs,
            num_sampling_times=1,
    ):
        """
        Forward pass for training and validation.
        median method does not support training, so it only returns 0.
        Args:
            inputs: Input data for training or validation.
            num_sampling_times: Number of sampling times for imputation.
        Returns:
            int: Returns 0 as median method does not support training.
        """
        
        return 0

    def compute_feature_medians(self, observed_data: torch.Tensor, conditional_mask: torch.Tensor):
        """
        Compute the mean for each feature across all valid entries.
        This function calvulates the mean of each feature across all time steps.
        Args:
            observed_data (torch.Tensor): Tensor of shape [B, C, T], where B is batch size, C is number of features, and T is number of time steps.
            conditional_mask (torch.Tensor): Tensor of shape [B, C, T], 1 if observed, 0 if missing
        Returns:
            torch.Tensor: Tensor of shape [C], mean for each feature over all valid entries.
        """
        C = observed_data.size(1)

        # merge B and to [B * T, C]
        data_flat = observed_data.permute(1, 0, 2).reshape(observed_data.size(1), -1)
        mask_flat = conditional_mask.permute(1, 0, 2).reshape(conditional_mask.size(1), -1)

        medians = [
            data_flat[i][mask_flat[i] == 1].median() if (mask_flat[i] == 1).any()
            else torch.tensor(0.0, device=observed_data.device)
            for i in range(C)
        ]
        return torch.stack(medians)  # shape: [C]

    
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

            samples = observed_data.clone()
            medians = self.compute_feature_medians(observed_data, cond_mask)
            for i in range(observed_data.size(1)):
                samples[:, i, :][cond_mask[:, i, :] == 0] = medians[i]
            
            samples = samples.unsqueeze(1)  # (B, 1, F, T) batch_size, num_sampling_times, num_features, num_steps

        return samples, observed_data, target_mask, observed_mask, observed_tp
