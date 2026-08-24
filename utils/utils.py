import numpy as np
import torch

def create_gt_mask(single_mask, missing_ratio=0.1, rng=None):
    # single_mask has shape (H, W, ...) and represents one sample's mask.
    flat_mask = single_mask.reshape(-1).copy()
    obs_indices = np.where(flat_mask)[0]
    miss_indices = rng.choice(
        obs_indices, int(len(obs_indices) * missing_ratio), replace=False
    )
    flat_mask[miss_indices] = False
    return flat_mask.reshape(single_mask.shape)

def sample_mask(
    observed_masks, 
    missing_ratio=0.05, 
    rng=None,
    missing_pattern='point'
):  # block missing, point missing
    if rng is None:
        random = np.random.random
        randint = np.random.randint
    else:
        random = rng.random
        randint = rng.integers

    shape = observed_masks.shape
    p_noise = missing_ratio

    if missing_pattern == 'time':
        batch_size, num_time, num_feature = shape
        mask_a = random((batch_size, num_time)) < p_noise
        mask = np.zeros(shape, dtype=bool)
        mask[mask_a, :] = True
    else:
        min_seq, max_seq = 12, 12 * 4
        p = missing_ratio if missing_pattern == 'block' else 0
        mask = random(shape) < p
        for col in range(mask.shape[1]):
            idxs = np.flatnonzero(mask[:, col])
            if not len(idxs):
                continue
            fault_len = min_seq
            if max_seq > min_seq:
                fault_len = fault_len + int(randint(max_seq - min_seq))
            idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
            idxs = np.unique(idxs_ext)
            idxs = np.clip(idxs, 0, shape[0] - 1)
            mask[idxs, col] = True
        mask = mask | (random(mask.shape) < p_noise)
    gt_masks = (1 - (mask | (1 - observed_masks)))
    return gt_masks.astype("uint8")


def data_normalize(observed_values, observed_masks, num_features):
    # calc mean and std and normalize values
    # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
    tmp_values = observed_values.reshape(-1, num_features)
    tmp_masks = observed_masks.reshape(-1, num_features)
    mean = np.zeros(num_features)
    std = np.zeros(num_features)
    for k in range(num_features):
        c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
        mean[k] = c_data.mean()
        std[k] = c_data.std()
    observed_values = ((observed_values - mean) / std * observed_masks)

    return observed_values 

class StandardScaler:
    def fit(self, x: np.ndarray):
        # x: shape = (N, C, T)
        N, C, T = x.shape
        x_flat = x.transpose(0, 2, 1).reshape(-1, C)  # (N*T, C)
        self.mean = np.mean(x_flat, axis=0)  # shape: (C,)
        self.std = np.std(x_flat, axis=0) + 1e-6      # shape: (C,)
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean[None, :, None]) / self.std[None, :, None]

    def inverse_transform(self, x: np.ndarray):
        return x * self.std[None, :, None] + self.mean[None, :, None]

    def fit_torch(self, x: torch.Tensor):
        # x: (B, C, T)
        x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # (B*T, C)
        self.mean = x_reshaped.mean(dim=0)  # (C,)
        self.std = x_reshaped.std(dim=0) + 1e-6
        return self

    def transform_torch(self, x: torch.Tensor):
        mean = torch.tensor(self.mean, dtype=torch.float32).to(x.device)
        std = torch.tensor(self.std, dtype=torch.float32).to(x.device)
        return (x - mean[None, :, None]) / std[None, :, None]

    def inverse_transform_torch(self, x: torch.Tensor, device):
        mean = torch.tensor(self.mean, dtype=torch.float32).to(device)
        std = torch.tensor(self.std, dtype=torch.float32).to(device)
        return x * std[None, :, None] + mean[None, :, None]


import numpy as np
import torch

class MaskedStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, values, masks):
        """
        values: shape (N, C, T)  (NumPy or torch.Tensor)
        masks:  same shape, 1 = observed, 0 = missing
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

        N, C, T = values.shape
        tmp_values = values.reshape(-1, C)     # (N*T, C)
        tmp_masks = masks.reshape(-1, C)       # (N*T, C)

        self.mean = np.zeros(C)
        self.std = np.zeros(C)

        for k in range(C):
            observed = tmp_values[:, k][tmp_masks[:, k] == 1]
            self.mean[k] = observed.mean()
            self.std[k] = observed.std() + 1e-6

        self.fitted = True
        return self

    def transform(self, values, masks):
        assert self.fitted, "Must fit before transform."
        if isinstance(values, torch.Tensor):
            values_np = values.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            scaled = ((values_np - self.mean[None, :, None]) / self.std[None, :, None]) * masks_np
            return torch.tensor(scaled, dtype=values.dtype).to(values.device)
        else:
            return ((values - self.mean[None, :, None]) / self.std[None, :, None]) * masks

    def inverse_transform(self, values, masks):
        assert self.fitted, "Must fit before inverse_transform."
        if isinstance(values, torch.Tensor):
            values_np = values.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            restored = (values_np * self.std[None, :, None] + self.mean[None, :, None]) * masks_np
            return torch.tensor(restored, dtype=values.dtype).to(values.device)
        else:
            return (values * self.std[None, :, None] + self.mean[None, :, None]) * masks

    def save(self, path):
        assert self.fitted, "Scaler not fitted yet."
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        stats = np.load(path)
        self.mean = stats['mean']
        self.std = stats['std']
        self.fitted = True
