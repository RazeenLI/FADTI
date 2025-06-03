import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.utils import sample_mask, data_normalize

def create_data():
    df = pd.read_csv(
        "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
        index_col="datetime",
        parse_dates=True,
    )
    df = df.sort_index()
    df['day'] = df.index.date

    # Drop test months: March, June, September, December
    df = df[~df.index.month.isin([3, 6, 9, 12])]

    feature_cols = df.columns.drop('day')
    grouped = df.groupby('day')
    observed_values = []

    for _, group in grouped:
        group = group.sort_index()
        arr = group[feature_cols].to_numpy()
        observed_values.append(arr)

    expected_time_steps = observed_values[0].shape[0]
    observed_values = [arr for arr in observed_values if arr.shape[0] == expected_time_steps]
    observed_values = np.stack(observed_values, axis=0)

    return observed_values

def get_data():
    # get total dataset, if not exit, create new dataset
    # 1. load data
    path = ("./data/ettm1.pk")

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

    # data normalization
    observed_values = np.nan_to_num(observed_values)
    # observed_values = data_normalize(observed_values, observed_masks, 7)
    # devide into three dataloader and return 
    dataset = ETT_Dataset(
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    indlist = np.arange(len(dataset))

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))
    # test_index = indlist[:2]
    # remain_index = indlist[2:]

    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_dataset = ETT_Dataset(
        use_index_list=train_index,
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = ETT_Dataset(
        use_index_list=valid_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = ETT_Dataset(
        use_index_list=test_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks,
        eval_length=num_steps
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    return train_loader, valid_loader, test_loader

class ETT_Dataset(Dataset):
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
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
