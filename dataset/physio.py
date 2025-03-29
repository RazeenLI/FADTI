# import pickle

# import os
# import re
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader, Dataset

# # 35 attributes which contains enough non-values
# attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
#               'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
#               'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
#               'Creatinine', 'ALP']


# def extract_hour(x):
#     h, _ = map(int, x.split(":"))
#     return h


# def parse_data(x):
#     # extract the last value for each attribute
#     x = x.set_index("Parameter").to_dict()["Value"]

#     values = []

#     for attr in attributes:
#         if x.__contains__(attr):
#             values.append(x[attr])
#         else:
#             values.append(np.nan)
#     return values


# def parse_id(id_, missing_ratio=0.1):
#     data = pd.read_csv("./data/physio/set-a/{}.txt".format(id_))
#     # set hour
#     data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

#     # create data for 48 hours x 35 attributes
#     observed_values = []
#     for h in range(48):
#         observed_values.append(parse_data(data[data["Time"] == h]))
#     observed_values = np.array(observed_values)
#     observed_masks = ~np.isnan(observed_values)

#     # randomly set some percentage as ground-truth
#     masks = observed_masks.reshape(-1).copy()
#     obs_indices = np.where(masks)[0].tolist()
#     miss_indices = np.random.choice(
#         obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
#     )
#     masks[miss_indices] = False
#     gt_masks = masks.reshape(observed_masks.shape)

#     observed_values = np.nan_to_num(observed_values)
#     observed_masks = observed_masks.astype("float32")
#     gt_masks = gt_masks.astype("float32")

#     return observed_values, observed_masks, gt_masks


# def get_idlist():
#     patient_id = []
#     for filename in os.listdir("./data/physio/set-a"):
#         match = re.search("\d{6}", filename)
#         if match:
#             patient_id.append(match.group())
#     patient_id = np.sort(patient_id)
#     return patient_id


# class Physio_Dataset(Dataset):
#     def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
#         self.eval_length = eval_length
#         np.random.seed(seed)  # seed for ground truth choice

#         self.observed_values = []
#         self.observed_masks = []
#         self.gt_masks = []
#         path = (
#             "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
#         )

#         if os.path.isfile(path) == False:  # if datasetfile is none, create
#             idlist = get_idlist()
#             for id_ in idlist:
#                 try:
#                     observed_values, observed_masks, gt_masks = parse_id(
#                         id_, missing_ratio
#                     )
#                     self.observed_values.append(observed_values)
#                     self.observed_masks.append(observed_masks)
#                     self.gt_masks.append(gt_masks)
#                 except Exception as e:
#                     print(id_, e)
#                     continue
#             self.observed_values = np.array(self.observed_values)
#             self.observed_masks = np.array(self.observed_masks)
#             self.gt_masks = np.array(self.gt_masks)

#             # calc mean and std and normalize values
#             # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
#             tmp_values = self.observed_values.reshape(-1, 35)
#             tmp_masks = self.observed_masks.reshape(-1, 35)
#             mean = np.zeros(35)
#             std = np.zeros(35)
#             for k in range(35):
#                 c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
#                 mean[k] = c_data.mean()
#                 std[k] = c_data.std()
#             self.observed_values = (
#                 (self.observed_values - mean) / std * self.observed_masks
#             )

#             with open(path, "wb") as f:
#                 pickle.dump(
#                     [self.observed_values, self.observed_masks, self.gt_masks], f
#                 )
#         else:  # load datasetfile
#             with open(path, "rb") as f:
#                 self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
#                     f
#                 )
#         if use_index_list is None:
#             self.use_index_list = np.arange(len(self.observed_values))
#         else:
#             self.use_index_list = use_index_list

#     def __getitem__(self, org_index):
#         index = self.use_index_list[org_index]
#         s = {
#             "observed_data": self.observed_values[index],
#             "observed_mask": self.observed_masks[index],
#             "gt_mask": self.gt_masks[index],
#             "timepoints": np.arange(self.eval_length),
#         }
#         return s

#     def __len__(self):
#         return len(self.use_index_list)


# def get_dataloader(
#         seed=1, 
#         nfold=None, 
#         batch_size=16, 
#         missing_ratio=0.1,
#         missing_pattern="point",
#     ):

#     # only to obtain total length of dataset
#     dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
#     indlist = np.arange(len(dataset))

#     np.random.seed(seed)
#     np.random.shuffle(indlist)

#     # 5-fold test
#     start = (int)(nfold * 0.2 * len(dataset))
#     end = (int)((nfold + 1) * 0.2 * len(dataset))
#     test_index = indlist[start:end]
#     remain_index = np.delete(indlist, np.arange(start, end))
#     # test_index = indlist[:2]
#     # remain_index = indlist[2:]


#     np.random.seed(seed)
#     np.random.shuffle(remain_index)
#     num_train = (int)(len(dataset) * 0.7)
#     train_index = remain_index[:num_train]
#     valid_index = remain_index[num_train:]

#     dataset = Physio_Dataset(
#         use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
#     )
#     train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
#     valid_dataset = Physio_Dataset(
#         use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
#     )
#     valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
#     test_dataset = Physio_Dataset(
#         use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
#     )
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
#     return train_loader, valid_loader, test_loader

import pickle

import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.utils import sample_mask, data_normalize

# 35 attributes which contains enough non-values
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']


def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    x = x.set_index("Parameter").to_dict()["Value"]
    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values

def get_idlist():
    patient_id = []
    for filename in os.listdir("./data/physio/set-a"):
        match = re.search("\d{6}", filename)
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id

def parse_id(id):
    data = pd.read_csv("./data/physio/set-a/{}.txt".format(id))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    for h in range(48):
        observed_values.append(parse_data(data[data["Time"] == h]))

    observed_values = np.array(observed_values)
    observed_values = np.nan_to_num(observed_values)
    return observed_values

def create_data():
    observed_values = []
    idlist = get_idlist()
    for id in idlist:
        try:
            observed_values.append(parse_id(id))
        except Exception as e:
            print(id, e)
            continue
    observed_values = np.array(observed_values)

    return observed_values 


def get_data():
    # get total dataset, if not exit, create new dataset
    # 1. load data
    path = ("./data/physio.pk")

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
        batch_size=16
):
    # get total dataset, if not exit, create new dataset
    observed_values = get_data()
    observed_masks = (~np.isnan(observed_values)).astype("uint8") # float32

    # add random mask
    rng = np.random.default_rng(seed)

    if missing_pattern == 'block':
        eval_masks = sample_mask(
            observed_masks=observed_masks,
            missing_ratio=missing_ratio, 
            rng=rng
        )
    elif missing_pattern == 'point':
        eval_masks = sample_mask(
            observed_masks=observed_masks,
            missing_ratio=missing_ratio, 
            rng=rng
        )
    gt_masks = (1 - (eval_masks | (1 - observed_masks))).astype('uint8')

    print(
        "Original missing ratio = {:.4f}\nArtificial missing ratio = {:.4f}\nArtificial missing pattern: {}\nOverall missing ratio = {:.4f}".format(
            1 - np.sum(observed_masks) / observed_masks.size,
            np.sum(eval_masks) / eval_masks.size,
            missing_pattern,
            1 - np.sum(gt_masks) / gt_masks.size,
        )
    )

    # data normalization
    observed_values = np.nan_to_num(observed_values)
    observed_values = data_normalize(observed_values, observed_masks, 35)
    # devide into three dataloader and return 
    dataset = Physio_Dataset(
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks
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

    train_dataset = Physio_Dataset(
        use_index_list=train_index,
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = Physio_Dataset(
        use_index_list=test_index, 
        observed_masks=observed_masks, 
        observed_values=observed_values, 
        gt_masks=gt_masks
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    return train_loader, valid_loader, test_loader

class Physio_Dataset(Dataset):
    def __init__(self, observed_values, observed_masks, gt_masks, eval_length=48, use_index_list=None):
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