import argparse
import torch
import datetime
import json
import yaml
import os
from model.process import Process
from torch.optim import AdamW, SGD, Adam
from utils.loaders import get_dataloader, get_model_optimizer


# Load Arguments
parser = argparse.ArgumentParser(description="model and data")

parser.add_argument(
    "--model",
    type=str,
    default="csdi",
    choices=[
        "mean", "median", "knn", "csdi", "csdi_ori", "fadti", "saits",
        "brits", "timesnet", "mtsci", "timemixer", "timemixerpp", "ssdts",
    ],
    help="Name of the model to use",
)
parser.add_argument("--data", type=str, default="ett", choices=["ett", "weather", "metr_la", "ecoli"], help="Dataset to use")
parser.add_argument('--device', default='cuda:0', help='Computation device')

parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--nfold", type=int, default=0, choices=range(5), help="Cross-validation fold (0-4)")
parser.add_argument("--missrate", type=float, default=0.1) # 0.1 0.5
parser.add_argument("--misspattern", type=str, default='point') # point block time

args = parser.parse_args()
print(args)

no_train_models = ["mean", "median", "knn"]

# Load model config
path = "config/" + args.model + ".yaml"
if args.model in no_train_models:
    path = "config/simple.yaml"
with open(path, "r") as f:
    config_model = yaml.safe_load(f)

# Load data config
path = "config/" + args.data + ".yaml"
with open(path, "r") as f:
    config_data = yaml.safe_load(f)

config = {**config_model, **config_data}

print(json.dumps(config, indent=4))

# Load Data
train_loader, valid_loader, test_loader, scaler = get_dataloader(args, config)
    
data_info = args.misspattern + "_" + str(args.missrate)

# Load Model
model, optimizer = get_model_optimizer(args, config)

model_process = Process(
        model=model,
        epochs=config["train"]["epochs"],
        save_strategy=config["train"]["save_strategy"],
        scaler=scaler,
        device=args.device, 
        optimizer=optimizer,           
)

# Create Save Place
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/" + args.model + "_" + args.data + "_" + data_info + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# train new model or load old model
if args.model not in no_train_models:
    if args.modelfolder == "":
        model_process.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model_process.model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# test
model_process.evaluate(
    test_loader=test_loader, 
    nsample=args.nsample, 
    scaler=1, 
    foldername=foldername
)
