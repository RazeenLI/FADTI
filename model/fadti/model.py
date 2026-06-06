import numpy as np
import torch
import torch.nn as nn

from .layers import DiffusionFADTI

from nn.process_data import get_process_data

class FADTI(nn.Module):
    def __init__(
            self,
            data_name,
            num_features,
            num_layers,
            num_heads,
            num_channels,
            num_diffusion_steps,
            num_steps,
            dim_time_embedding,
            dim_feature_embedding,
            dim_diffusion_embedding,
            is_unconditional,
            schedule,
            beta_start,
            beta_end,
            target_strategy,
            method, # "dft" "stft" "frsst"
            type_layer, # "attn" "conv" "atten+conv"
            device,
    ):
        super().__init__()
        self.device = device

        self.process_data = get_process_data(data_name)

        # self.dim_target = dim_target # target embedding dim
        self.dim_time_embedding = dim_time_embedding # time embedding dim
        self.dim_feature_embedding = dim_feature_embedding # feature embedding dim
        
        self.is_unconditional = is_unconditional
        self.num_features = num_features
        self.num_channels = num_channels
        self.num_diffusion_steps = num_diffusion_steps

        # Side Information (Conditional Information)
        dim_side = dim_time_embedding + dim_feature_embedding
        if self.is_unconditional:
            dim_input = 1
        else:
            dim_input = 2
            dim_side += 1 # for conditional mask


        # Feature Embedding
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_features,
            embedding_dim=dim_feature_embedding,
        )

        self.diffussion_model = DiffusionFADTI(
            num_diffusion_steps=num_diffusion_steps,
            dim_diffusion_embedding=dim_diffusion_embedding,
            dim_input=dim_input,
            dim_side=dim_side,
            num_channels=num_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            num_steps=num_steps,
            method=method,
            type_layer=type_layer,
        )

        # Noise Paremeters for diffusion model
        if schedule == "quad":
            # square toot then square
            self.beta = np.linspace(beta_start**0.5, beta_end**0.5, self.num_diffusion_steps) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.num_diffusion_steps)
        else:
            raise ValueError(f"The argument schedule should be 'quad' or 'linear', but got {schedule}")
        
        self.alpha_hat = 1 - self.beta
        # cumulative product for reduction or reverse calculations
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer(
            "alpha_torch", 
            torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(
            self, 
            pos, #(batch_size, num_steps)
            d_model=128
    ):
        """ transformer positional encoding """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe # (batch_size, num_steps, d_model)

    def get_side_info(
            self, 
            observed_tp, 
            conditional_mask
    ):
        batch_size, num_features, num_steps = conditional_mask.shape
        # device = observed_tp.device
        # Time Embedding
        time_embedding = self.time_embedding(
            pos=observed_tp, 
            d_model=self.dim_time_embedding
        )  # (batch_size, num_steps, dim_time_embedding)
        time_embedding = time_embedding.to(self.device)
        time_embedding = time_embedding.unsqueeze(2).expand(-1, -1, num_features, -1) # (batch_size, num_steps, num_features, dim_time_embedding)
        # Feature Embedding
        feature_embedding = self.embedding_layer(torch.arange(self.num_features).to(self.device))  # (num_feature, dim_feature_embedding)
        feature_embedding = feature_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, num_steps, -1, -1)  # (batch_size, num_steps, num_features, dim_feature_embedding)

        side_info = torch.cat([time_embedding, feature_embedding], dim=-1)  # (batch_size, num_steps, num_features, dim_time_embedding+dim_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (batch_size, *, num_features, num_steps)

        if not self.is_unconditional:
            side_mask = conditional_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info
    
    def impute(
            self,
            observed_data,
            conditional_mask,
            side_info,
            num_sampling_times, # sampleing times, num of imputed samples
    ):
        batch_size, num_features, num_steps = observed_data.shape
        # device = observed_data.device
        imputed_samples = torch.zeros(batch_size, num_sampling_times, num_features, num_steps).to(self.device)

        for i in range(num_sampling_times):
            if self.is_unconditional:
                # generate noisy observation for unconditional model
                # add noise step by step
                noisy_observation = observed_data
                noisy_conditional_history = []
                for t in range(self.num_diffusion_steps):
                    noise = torch.randn_like(noisy_observation)
                    noisy_observation = (self.alpha_hat[t] ** 0.5) * noisy_observation + self.beta[t] ** 0.5 * noise
                    noisy_conditional_history.append(noisy_observation * conditional_mask)
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_diffusion_steps - 1, -1, -1):
                if self.is_unconditional:
                    diffusion_input = conditional_mask * noisy_conditional_history[t] + (1.0 - conditional_mask) * current_sample
                    diffusion_input = diffusion_input.unsqueeze(1)  # (batch_size, 1, num_features, num_steps)
                else:
                    cond_obs = (conditional_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - conditional_mask) * current_sample).unsqueeze(1)
                    diffusion_input = torch.cat([cond_obs, noisy_target], dim=1)  # (batch_size, 2, num_features, num_steps)

                predicted = self.diffussion_model(diffusion_input, side_info, torch.tensor([t]).to(self.device))

                scaling_factor1 = 1 / self.alpha_hat[t] ** 0.5
                scaling_factor2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = scaling_factor1 * (current_sample - scaling_factor2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    
    def set_input_to_diffusion_model(
            self,
            noisy_data,
            observed_data,
            conditional_mask,
    ):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (conditional_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - conditional_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def calc_loss_valid(
        self, 
        observed_data, 
        conditional_mask, 
        observed_mask, 
        side_info,
    ):
        loss_sum = 0
        for step in range(self.num_diffusion_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, 
                conditional_mask, 
                observed_mask, 
                side_info, 
                set_step=step
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_diffusion_steps
        
    def calc_loss(
        self, 
        observed_data, 
        conditional_mask, 
        observed_mask, 
        side_info, 
        noise_type="gaussian",
        set_step=-1
    ):
        batch_size, num_features, num_steps = observed_data.shape
        # device = observed_data.device

        # diffusion step
        if not self.training:  # for validation
            step = (torch.ones(batch_size) * set_step).long().to(self.device)
        else:
            step = torch.randint(0, self.num_diffusion_steps, [batch_size]).to(self.device)

        current_alpha = self.alpha_torch[step]  # (batch_size, 1, 1)
        if noise_type == "gaussian":
            noise = torch.randn_like(observed_data)
        elif noise_type == "laplace":
            laplace = torch.distributions.Laplace(
                loc=torch.zeros_like(observed_data),
                scale=torch.ones_like(observed_data) / (2 ** 0.5)  # 方差≈1
            )
            noise = laplace.sample()
        else: # uniform noise
            noise = torch.rand_like(observed_data) * (2 * (3 ** 0.5)) - (3 ** 0.5)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffusion_model(noisy_data, observed_data, conditional_mask) # (batch_size, 2|1, num_features, num_steps)

        predicted = self.diffussion_model(total_input, side_info, step)  # (batch_size, num_features, num_steps)

        target_mask = observed_mask - conditional_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
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

            conditional_mask = self.get_randmask(observed_mask)
            side_info = self.get_side_info(observed_tp, conditional_mask)

            # training loss
            results["loss"] = self.calc_loss(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
                side_info=side_info, 
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

            side_info = self.get_side_info(observed_tp, conditional_mask)
            # validating loss
            results["loss"] = self.calc_loss_valid(
                observed_data=observed_data, 
                conditional_mask=conditional_mask, 
                observed_mask=observed_mask, 
                side_info=side_info, 
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

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, num_sampling_times)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp
    
        # side_info = self.get_side_info(observed_tp, conditional_mask)
        # samples = self.impute(
        #     observed_data=observed_data,
        #     conditional_mask=conditional_mask,
        #     side_info=side_info,
        #     num_sampling_times=num_sampling_times
        # ) # (batch_size, num_sampling_times, num_features, num_steps)
        # repeated_observation = observed_data.unsqueeze(1).repeat(1, num_sampling_times, 1, 1)
        # repeated_mask = conditional_mask.unsqueeze(1).repeat(1, num_sampling_times, 1, 1)
        # imputed_data = repeated_observation + samples * (1 - repeated_mask)

        # results["imputed_data"] = imputed_data.permute(0, 1, 3, 2)  # (batch_size, num_sampling_times, num_steps, num_features)
    
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
    