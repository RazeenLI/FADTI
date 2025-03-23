import numpy as np

def sample_mask(
    shape, 
    p=0.0015, 
    p_noise=0.05, 
    max_seq=1, 
    min_seq=1, 
    rng=None
):  # block missing, point missing
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
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
    mask = mask | (rand(mask.shape) < p_noise)
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