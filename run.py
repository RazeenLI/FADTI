# python run.py --model "ftcsdi" --data "ett" --nsample 100 --nfold 0 --missrate 0.1 --misspattern "point" --device "cuda:6" --modelfolder "ftcsdi_ett_point_0.1_20250404_085422"
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


# Load Arguments
parser = argparse.ArgumentParser(description="model and data")

parser.add_argument("--model", type=str, default="csdi", help="Name of the model to use")
parser.add_argument("--data", type=str, default="physio", help="Name of the dataset to use")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')

parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--nfold", type=int, default=0) # for 5fold test (valid value:[0-4])
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
data_info = "None"
if args.data == "physio":
    from dataset.physio import get_dataloader
    train_loader, valid_loader, test_loader = get_dataloader(
        seed=config["data"]["seed"],
        nfold=args.nfold,
        batch_size=config["data"]["batch_size"],
        missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
        missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
        num_steps=config["data"]["num_steps"],
    )
elif args.data == "ett":
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
if args.model == "csdi":
    from model.csdi.model import CSDI
    config_diff = config["diffusion"]
    model = CSDI(
            data_name=args.data,
            num_features=config["data"]["num_features"],
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
            device=args.device,
        )
elif args.model == "csdi_ori":
    from model.csdi_ori.model import CSDI_base
    config_diff = config["diffusion"]
    model = CSDI_base(
            data_name=args.data,
            num_features=config["data"]["num_features"],
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
            device=args.device,
        )
elif args.model == "ftcsdi":
    from model.fourier_t_csdi.model import FTCSDI
    config_diff = config["diffusion"]
    model = FTCSDI(
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
            device=args.device,
        )
elif args.model == "saits":
    from model.saits.model import SAITS
    config_diff = config["diffusion"]
    model = SAITS(
            data_name=args.data,
            n_groups=config["model"]['n_groups'],
            n_group_inner_layers=config["model"]['n_group_inner_layers'],
            dim_time=config["data"]["num_steps"],#config["model"]['n_groups'],
            dim_feature=config["data"]["num_features"],#=config["model"]['n_groups'],
            dim_model=config["model"]['dim_model'],
            dim_hidden=config["model"]['dim_hidden'],
            num_heads=config["model"]['num_heads'], # num_heads
            dim_k=config["model"]['dim_k'],
            dim_v=config["model"]['dim_v'],
            dropout=config["model"]['dropout'],
            reconstruction_loss_weight=config["model"]['reconstruction_loss_weight'],
            imputation_loss_weight=config["model"]['imputation_loss_weight'],
            diagonal_attention_mask=config["model"]['diagonal_attention_mask'],
            param_sharing_strategy=config["model"]['param_sharing_strategy'],
            input_with_mask=config["model"]['input_with_mask'],
            MIT=config["model"]['MIT'],
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

