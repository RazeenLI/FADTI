# python run.py --model "fadti" --ffttype "fbm" --timetype "attn" --data "ett" --nsample 100 --nfold 0 --missrate 0.1 --misspattern "point" --device "cuda:6" --modelfolder "ftcsdi_ett_point_0.1_20250404_085422"
# python run.py --model "saits" --data "ett" --nsample 100 --nfold 0 --missrate 0.1 --misspattern "point" --device "cuda:6" --modelfolder "saits_ett_time_0.5_20250410_103809"
# python run.py --model "csdi" --data "ett" --nsample 100 --nfold 0 --device "cuda:5"
# python run.py --model "csdi_ori" --data "ett" --nsample 100 --nfold 0 --device "cuda:4" 
import argparse
import torch
import datetime
import json
import yaml
import os
from model.process import Process
from model.fadti.model import FADTI


# Load Arguments
parser = argparse.ArgumentParser(description="model and data")

parser.add_argument("--model", type=str, default="csdi", help="Name of the model to use")
parser.add_argument("--ffttype", type=str, default="fbm", help="Name of the model to use") # frsst fsst
parser.add_argument("--timetype", type=str, default="attn", help="Name of the model to use") # attn conv attn+conv
parser.add_argument("--model", type=str, default="csdi", help="Name of the model to use")

parser.add_argument("--data", type=str, default="physio", help="Name of the dataset to use")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--nfold", type=int, default=0) # for 5fold test (valid value:[0-4])
parser.add_argument("--missrate", type=float, default=0.1) # 0.1 0.5
parser.add_argument("--misspattern", type=str, default='point') # point block time
# parser.add_argument("--method", type=str, default='point') # point block time

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
data_info = "None"
if args.data == "ett":
    from dataset.ett import get_dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=config["data"]["seed"],
        nfold=args.nfold,
        batch_size=config["data"]["batch_size"],
        missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
        missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
        num_steps=config["data"]["num_steps"],
    )
elif args.data == "weather":
    from dataset.weather import get_dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=config["data"]["seed"],
        nfold=args.nfold,
        batch_size=config["data"]["batch_size"],
        missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
        missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
        num_steps=config["data"]["num_steps"],
    )
    
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

model_process = Process(
        model=model,
        learning_rate=config["train"]["lr"],
        epochs=config["train"]["epochs"],
        save_strategy=config["train"]["save_strategy"],
        device=args.device,            
)

# Create Save Place
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/" + args.model + "_" + args.data + "_" + data_info + "_" + current_time + "/"
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

