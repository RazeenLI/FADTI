"""SSD-TS wrapper adapted to this repository's unified interface.

Place this file at: model/ssdts/model.py
It follows the same public API as your other models:
    loss = model(batch)
    samples, target, eval_mask, observed_mask, observed_tp = model.evaluate(batch, nsample)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from nn.process_data import get_process_data
from .layers import BiSSM2Imputer


class SSDTS(nn.Module):
    def __init__(
        self,
        data_name: str,
        num_features: int,
        num_steps: int,
        layers: int,
        seq_dim: int,
        res_channels: int,
        diffusion_embedding_dim: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
        num_ssm: int,
        cond_ssm_num: int,
        input_ssm_num: int,
        num_ch: int,
        expand_c: int,
        expand_s: int,
        headdim_c: int,
        headdim_s: int,
        device: str | torch.device,
        only_generate_missing: bool = True,
        valid_all_steps: bool = False,
    ):
        super().__init__()
        self.process_data = get_process_data(data_name)
        self.device = torch.device(device)
        self.num_features = num_features
        self.num_steps = num_steps
        self.diffusion_steps = diffusion_steps
        self.only_generate_missing = only_generate_missing
        self.valid_all_steps = valid_all_steps

        self.diffmodel = BiSSM2Imputer(
            layers=layers,
            seq_len=num_steps,
            seq_dim=seq_dim,
            in_channels=num_features,
            res_channels=res_channels,
            diffusion_embedding_dim=diffusion_embedding_dim,
            num_steps=diffusion_steps,
            num_ssm=num_ssm,
            cond_ssm_num=cond_ssm_num,
            input_ssm_num=input_ssm_num,
            num_ch=num_ch,
            expand_c=expand_c,
            expand_s=expand_s,
            headdim_c=headdim_c,
            headdim_s=headdim_s,
        )

        if schedule == "quad":
            beta = np.linspace(beta_start ** 0.5, beta_end ** 0.5, diffusion_steps) ** 2
        elif schedule == "linear":
            beta = np.linspace(beta_start, beta_end, diffusion_steps)
        else:
            raise ValueError(f"Unsupported diffusion schedule: {schedule}")

        beta = torch.tensor(beta, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        beta_tilde = beta.clone()
        beta_tilde[1:] = beta[1:] * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sigma", torch.sqrt(beta_tilde))

    def _unpack_batch(self, batch):
        res = self.process_data(batch, self.device)
        observed_data = res["observed_data"]      # (B, K, L)
        observed_mask = res["observed_mask"]      # natural observed points
        observed_tp = res["observed_tp"]
        cond_mask = res["gt_mask"]                # conditional points kept visible
        return observed_data, observed_mask, observed_tp, cond_mask

    def _loss_at_t(
        self,
        observed_data: torch.Tensor,
        observed_mask: torch.Tensor,
        cond_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        target_mask = (observed_mask - cond_mask).clamp(min=0.0)
        noise = torch.randn_like(observed_data)

        if self.only_generate_missing:
            # Original SSD-TS preserves conditional points in x_t and computes loss on missing targets only.
            diffusion_noise = observed_data * cond_mask + noise * (1.0 - cond_mask)
        else:
            diffusion_noise = noise

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        noisy_data = torch.sqrt(alpha_bar_t) * observed_data + torch.sqrt(1.0 - alpha_bar_t) * diffusion_noise
        pred_noise = self.diffmodel((noisy_data, observed_data, cond_mask, t.view(-1, 1)))

        denom = target_mask.sum()
        if denom.item() == 0:
            return pred_noise.sum() * 0.0
        return (((pred_noise - diffusion_noise) ** 2) * target_mask).sum() / denom

    def forward(self, batch) -> torch.Tensor:
        observed_data, observed_mask, _observed_tp, cond_mask = self._unpack_batch(batch)
        B = observed_data.shape[0]

        if (not self.training) and self.valid_all_steps:
            loss = 0.0
            for step in range(self.diffusion_steps):
                t = torch.full((B,), step, device=observed_data.device, dtype=torch.long)
                loss = loss + self._loss_at_t(observed_data, observed_mask, cond_mask, t)
            return loss / self.diffusion_steps

        t = torch.randint(0, self.diffusion_steps, (B,), device=observed_data.device, dtype=torch.long)
        return self._loss_at_t(observed_data, observed_mask, cond_mask, t)

    @torch.no_grad()
    def impute_once(self, observed_data: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        current_sample = torch.randn_like(observed_data)
        B = observed_data.shape[0]

        for step in range(self.diffusion_steps - 1, -1, -1):
            if self.only_generate_missing:
                current_sample = current_sample * (1.0 - cond_mask) + observed_data * cond_mask

            t = torch.full((B,), step, device=observed_data.device, dtype=torch.long)
            pred_noise = self.diffmodel((current_sample, observed_data, cond_mask, t.view(-1, 1)))

            current_sample = (
                current_sample
                - (1.0 - self.alpha[step]) / torch.sqrt(1.0 - self.alpha_bar[step]) * pred_noise
            ) / torch.sqrt(self.alpha[step])

            if step > 0:
                current_sample = current_sample + self.sigma[step] * torch.randn_like(current_sample)

        if self.only_generate_missing:
            current_sample = current_sample * (1.0 - cond_mask) + observed_data * cond_mask
        return current_sample

    @torch.no_grad()
    def evaluate(self, batch, n_samples: int):
        observed_data, observed_mask, observed_tp, cond_mask = self._unpack_batch(batch)
        eval_mask = (observed_mask - cond_mask).clamp(min=0.0)
        samples = torch.zeros(
            observed_data.shape[0],
            n_samples,
            observed_data.shape[1],
            observed_data.shape[2],
            device=observed_data.device,
        )
        for i in range(n_samples):
            samples[:, i] = self.impute_once(observed_data, cond_mask)
        return samples, observed_data, eval_mask, observed_mask, observed_tp
