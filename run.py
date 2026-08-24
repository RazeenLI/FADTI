import argparse
import torch
import datetime
import json
import yaml
import os
from model.process import Process
from model.fadti.model import FADTI
from torch.optim import AdamW

from utils.loaders import get_dataloader


# Load Arguments
parser = argparse.ArgumentParser(description="model and data")

parser.add_argument("--model", type=str, default="fadti", choices=["fadti"], help="Model to use")
parser.add_argument("--ffttype", type=str, default="dft", choices=["none", "dft", "stft", "frsst"], help="Fourier transform type")
parser.add_argument("--timetype", type=str, default="attn", choices=["attn", "conv"], help="Time-processing layer")

parser.add_argument("--data", type=str, default="ett", choices=["ett", "weather", "metr_la", "yeast"], help="Dataset to use")
parser.add_argument('--device', default='cuda:0', help='Computation device')

parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--nfold", type=int, default=0, choices=range(5), help="Cross-validation fold (0-4)")
parser.add_argument("--missrate", type=float, default=0.1) # 0.1 0.5
parser.add_argument("--misspattern", type=str, default='point') # point block time

args = parser.parse_args()
print(args)

# Load model config
path = "config/" + args.model + ".yaml"

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
config_diff = config["diffusion"]
model = FADTI(
        data_name=args.data,
        num_features=config["data"]["num_features"],
        num_steps=config["data"]["num_steps"],
        num_layers=config_diff["layers"],
        num_heads=config_diff["nheads"],
        num_channels=config_diff["channels"],
        num_diffusion_steps=config_diff["num_steps"],
        dim_time_embedding=config["model"]["timeemb"],
        dim_feature_embedding=config["model"]["featureemb"],
        dim_diffusion_embedding=config_diff["diffusion_embedding_dim"],
        is_unconditional=config["model"]["is_unconditional"],
        schedule=config_diff["schedule"],
        beta_start=config_diff["beta_start"],
        beta_end=config_diff["beta_end"],
        target_strategy=config["model"]["target_strategy"],
        method=args.ffttype, # "fbm", "frsst"
        type_layer=args.timetype, # "attn" "conv" "atten+conv"
        device=args.device,
    )

optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)

model_info = args.ffttype + "_" + args.timetype

# Load Model


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
foldername = "./save/" + args.model + "_" + model_info + "_" + args.data + "_" + data_info + "_" + str(args.nsample) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# train new model or load old model
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
