import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.utils import sample_mask, data_normalize, MaskedStandardScaler

import numpy as np
import pandas as pd

def create_data():
    df = pd.read_hdf("./data/metr_la/metr-la.h5")

    data = df.values  # shape: (34272, 207)
    data[data == 0.0] = np.nan

    # One point per hour: 5 minutes × 12 = 1 hour
    interval = 12  # 12 time steps = 1 hour
    hourly_data = data[::interval]  # shape: [T_total // 12, 207]

    # 24 hours a day, only retain the full day's data
    day_len = 24
    T_total = hourly_data.shape[0]
    num_full_days = T_total // day_len
    hourly_data = hourly_data[:num_full_days * day_len]  # Remove redundant time steps
    hourly_data = hourly_data.reshape(num_full_days, day_len, -1)  # shape: [num_days, 24, 207]

    # No longer group by sensor, keep all
    # hourly_data = hourly_data.reshape(-1, 24, 207)

    return hourly_data

def get_data():
    # get total dataset, if not exit, create new dataset
    # 1. load data
    path = ("./data/metrla.pk")

    if os.path.isfile(path):  
        # load data file
        with open(path, "rb") as f:
            observed_values = pickle.load(f)
        return observed_values
    else:
        observed_values = create_data()
        # save data
        with open(path, "wb") as f:
            pickle.dump(observed_values, f)
    return observed_values
    
    
    
def get_dataloader(
        nfold=None,
        seed=1,
        missing_pattern='block',
        missing_ratio=0.0,
        batch_size=16,
        num_steps=96
):
    # get total dataset, if not exit, create new dataset
    observed_values = get_data()
    observed_masks = (~np.isnan(observed_values)).astype("uint8") # float32

    # add random mask
    rng = np.random.default_rng(seed)

    gt_masks = sample_mask(
        observed_masks=observed_masks,
        missing_ratio=missing_ratio, 
        rng=rng,
        missing_pattern=missing_pattern
    )
    # gt_masks = (1 - (gt_masks | (1 - observed_masks))).astype('uint8')

    print(
        "Original missing ratio = {:.4f}\nArtificial missing pattern: {}\nOverall missing ratio = {:.4f}".format(
            1 - np.sum(observed_masks) / observed_masks.size,
            missing_pattern,
            1 - np.sum(gt_masks) / gt_masks.size,
        )
    )

    indlist = np.arange(len(observed_values))

    # 5-fold test
    start = (int)(nfold * 0.2 * len(indlist))
    end = (int)((nfold + 1) * 0.2 * len(indlist))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    # test_index = indlist[:2]
    # remain_index = indlist[2:]

    num_train = (int)(len(indlist) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_data = observed_values[train_index]
    train_masks = observed_masks[train_index]

    scaler = MaskedStandardScaler().fit(train_data, train_masks)
    observed_values = scaler.transform(observed_values, observed_masks)
    observed_values = np.nan_to_num(observed_values)

    train_dataset = MetrLA_Dataset(
        use_index_list=train_index,
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = MetrLA_Dataset(
        use_index_list=valid_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = MetrLA_Dataset(
        use_index_list=test_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    return train_loader, valid_loader, test_loader, scaler

class MetrLA_Dataset(Dataset):
    def __init__(self, observed_values, observed_masks, gt_masks, eval_length, use_index_list=None):
        self.eval_length = eval_length
        self.observed_values = observed_values
        self.observed_masks = observed_masks
        self.gt_masks = gt_masks
        self.use_index_list = np.arange(len(self.observed_values)) if use_index_list is None else use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "next_data": self.observed_values[index + 1] if (index + 1 in self.use_index_list) else np.zeros_like(self.observed_values[index])  # 占位张量
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    

def get_dataset(
        nfold=None,
        seed=1,
        missing_pattern='block',
        missing_ratio=0.0,
        batch_size=16,
        num_steps=96
):
    # get total dataset, if not exit, create new dataset
    observed_values = get_data()
    observed_masks = (~np.isnan(observed_values)).astype("uint8") # float32

    # add random mask
    rng = np.random.default_rng(seed)

    gt_masks = sample_mask(
        observed_masks=observed_masks,
        missing_ratio=missing_ratio, 
        rng=rng,
        missing_pattern=missing_pattern
    )
    # gt_masks = (1 - (gt_masks | (1 - observed_masks))).astype('uint8')

    print(
        "Original missing ratio = {:.4f}\nArtificial missing pattern: {}\nOverall missing ratio = {:.4f}".format(
            1 - np.sum(observed_masks) / observed_masks.size,
            missing_pattern,
            1 - np.sum(gt_masks) / gt_masks.size,
        )
    )

    indlist = np.arange(len(observed_values))

    # 5-fold test
    start = (int)(nfold * 0.2 * len(indlist))
    end = (int)((nfold + 1) * 0.2 * len(indlist))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    # test_index = indlist[:2]
    # remain_index = indlist[2:]

    num_train = (int)(len(indlist) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_data = observed_values[train_index]
    train_masks = observed_masks[train_index]

    scaler = MaskedStandardScaler().fit(train_data, train_masks)
    observed_values = scaler.transform(observed_values, observed_masks)
    observed_values = np.nan_to_num(observed_values)

    X_ori = observed_values[train_index]
    gt_mask = gt_masks[train_index]
    X = X_ori.copy()
    X[gt_mask == 1] = np.nan
    train_set = {
        'X': X, 
        'X_ori': X_ori,
        'mask_X_gt': gt_mask
    }

    X_ori = observed_values[valid_index]
    gt_mask = gt_masks[valid_index]
    X = X_ori.copy()
    X[gt_mask == 1] = np.nan
    valid_set = {
        'X': X,
        'X_ori': X_ori, 
        'mask_X_gt': gt_mask
    }

    X_ori = observed_values[test_index]
    gt_mask = gt_masks[test_index]
    X = X_ori.copy()
    X[gt_mask == 1] = np.nan
    test_set = {
        'X': X,
        'X_ori': X_ori,
        'mask_X_gt': gt_mask
    }

    return train_set, valid_set, test_set, scaler