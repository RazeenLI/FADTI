# python check.py --device "cuda:6"
import argparse
import torch
import datetime
import json
import yaml
import os
from model.process import Process


# Load Arguments
parser = argparse.ArgumentParser(description="model and data")

parser.add_argument("--data", type=str, default="physio", help="Name of the dataset to use")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

# parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

# parser.add_argument("--seed", type=int, default=1)
# parser.add_argument("--testmissingratio", type=float, default=0.1)
# parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")

args = parser.parse_args()
print(args)

# Load data config
path = "config/" + args.data + ".yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

# config["model"]["is_unconditional"] = args.unconditional
# config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

from dataset.physio import get_dataloader
train_loader_1, valid_loader_1, test_loader_1 = get_dataloader(
    seed=config["data"]["seed"],
    nfold=config["data"]["nfold"],
    batch_size=config["data"]["batch_size"],
    missing_ratio=config["data"]["test_missing_ratio"],
    missing_pattern=config["data"]["missing_pattern"],
)
data_info = config["data"]["missing_pattern"] + "_" + str(config["data"]["test_missing_ratio"])

# from dataset.physio_old import get_dataloader
# train_loader_2, valid_loader_2, test_loader_2 = get_dataloader(
#     seed=config["data"]["seed"],
#     nfold=config["data"]["nfold"],
#     batch_size=config["data"]["batch_size"],
#     missing_ratio=config["data"]["test_missing_ratio"],
#     missing_pattern=config["data"]["missing_pattern"],
# )


train_loader_3, valid_loader_3, test_loader_3 = get_dataloader(
    seed=config["data"]["seed"],
    nfold=config["data"]["nfold"],
    batch_size=config["data"]["batch_size"],
    missing_ratio=config["data"]["test_missing_ratio"],
    missing_pattern=config["data"]["missing_pattern"],
)

all_equal = True
for i, (batch1, batch2) in enumerate(zip(test_loader_1, test_loader_3)):
    for key in batch1.keys():
        if not torch.equal(batch1[key], batch2[key]):
            print(f"Batch {i}, key '{key}' is different.")
            all_equal = False
for i, (batch1, batch2) in enumerate(zip(valid_loader_1, valid_loader_3)):
    for key in batch1.keys():
        if not torch.equal(batch1[key], batch2[key]):
            print(f"Batch {i}, key '{key}' is different.")
            all_equal = False
if all_equal:
    print("All batches are identical.")
