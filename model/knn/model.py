import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import KNNImputer

from nn.process_data import get_process_data

class KNN(nn.Module):
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
        KNN method does not support training, so it only returns 0.
        Args:
            inputs: Input data for training or validation.
            num_sampling_times: Number of sampling times for imputation.
        Returns:
            int: Returns 0 as mean method does not support training.
        """
        
        return 0

    def compute_feature_knn(self, observed_data: torch.Tensor, conditional_mask: torch.Tensor, k: int = 5):
        """
        Apply KNN imputation across batch dimension (feature-wise).
        Args:
            observed_data: Tensor of shape [B, C, T]
            conditional_mask: Tensor of shape [B, C, T], 1 if observed, 0 if missing
            k: number of neighbors
        Returns:
            imputed_data: Tensor of shape [B, C, T] with missing entries filled
        """
        B, C, T = observed_data.shape
        imputed_data = observed_data.clone()

        for i in range(T):
            data_step = observed_data[:, :, i].cpu().numpy()  # shape: [B, C]
            mask_step = conditional_mask[:, :, i].cpu().numpy()

            data_step[mask_step == 0] = np.nan  # Mark missing values.

            nan_columns = np.all(np.isnan(data_step), axis=0)  # shape: [C]
            data_step[:, nan_columns] = 0.0  # Use zero when an entire column is missing.

            knn = KNNImputer(n_neighbors=k)
            imputed_step = knn.fit_transform(data_step)  # shape: [B, C]

            imputed_data[:, :, i] = torch.tensor(imputed_step, device=observed_data.device)

        return imputed_data

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

            samples = self.compute_feature_knn(observed_data, cond_mask, k=5)  # or k=3
            
            samples = samples.unsqueeze(1)  # (B, 1, F, T) batch_size, num_sampling_times, num_features, num_steps

        return samples, observed_data, target_mask, observed_mask, observed_tp
