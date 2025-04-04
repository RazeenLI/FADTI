import numpy as np

def create_gt_mask(single_mask, missing_ratio=0.1, rng=None):
    # single_mask 的 shape 为 (H, W, ...) 表示单个数据的掩码
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
    rng=None
):  # block missing, point missing
    if rng is None:
        rng = np.random

    # # 假设 observed_masks 的第一维代表不同的数据样本，shape = (N, ...)
    # masks_list = []
    # for i in range(observed_masks.shape[0]):
    #     masks_list.append(create_gt_mask(observed_masks[i], missing_ratio=missing_ratio, rng=rng))

    # masks = np.stack(masks_list, axis=0)

    # return masks.astype("uint8")

    shape = observed_masks.shape
    min_seq, max_seq = 12, 12 * 4
    p = 0
    p_noise = missing_ratio
    mask = rng.random(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(rng.randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rng.random(mask.shape) < p_noise)
    return mask.astype("uint8")


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