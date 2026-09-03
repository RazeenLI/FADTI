from torch.optim import AdamW, Adam


def get_dataloader(args, config):
    if args.data == "ett":
        from dataset.ett import get_dataloader
        train_loader, valid_loader, test_loader, scaler = get_dataloader(
            seed=config["data"]["seed"],
            nfold=args.nfold,
            batch_size=config["data"]["batch_size"],
            missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
            missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
            num_steps=config["data"]["num_steps"],
        )
    elif args.data == "weather":
        from dataset.weather import get_dataloader
        train_loader, valid_loader, test_loader, scaler = get_dataloader(
            seed=config["data"]["seed"],
            nfold=args.nfold,
            batch_size=config["data"]["batch_size"],
            missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
            missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
            num_steps=config["data"]["num_steps"],
        )
    elif args.data == "metr_la":
        from dataset.metr_la import get_dataloader
        train_loader, valid_loader, test_loader, scaler = get_dataloader(
            seed=config["data"]["seed"],
            nfold=args.nfold,
            batch_size=config["data"]["batch_size"],
            missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
            missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
            num_steps=config["data"]["num_steps"],
        )
    elif args.data == "ecoli":
        from dataset.ecoli import get_dataloader
        train_loader, valid_loader, test_loader, scaler = get_dataloader(
            seed=config["data"]["seed"],
            nfold=args.nfold,
            batch_size=config["data"]["batch_size"],
            missing_ratio=args.missrate, # config["data"]["test_missing_ratio"],
            missing_pattern=args.misspattern, # config["data"]["missing_pattern"],
            num_steps=config["data"]["num_steps"],
        )
    return train_loader, valid_loader, test_loader, scaler

def get_model_optimizer(args, config):
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
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
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
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "fadti":
        from model.fadti.model import FADTI
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
            method="dft", # "fsst", "frsst"
            type_layer="attn", # "attn" "conv" "atten+conv"
            device=args.device,
        )
        optimizer = AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
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
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "mean":
        from model.mean.model import Mean
        config_diff = config["diffusion"]
        model = Mean(
            data_name=args.data,
            device=args.device,
        )
        optimizer = None
    elif args.model == "median":
        from model.median.model import Median
        config_diff = config["diffusion"]
        model = Median(
            data_name=args.data,
            device=args.device,
        )
        optimizer = None
    elif args.model == "knn":
        from model.knn.model import KNN
        config_diff = config["diffusion"]
        model = KNN(
            data_name=args.data,
            device=args.device,
        )
        optimizer = None
    elif args.model == "brits":
        from model.brits.model import BRITS
        # config_diff = config["diffusion"]
        model = BRITS(
            data_name=args.data,
            num_features=config["data"]["num_features"],
            num_channels=config["model"]["channels"],
            num_steps=config["data"]["num_steps"],
            device=args.device,
        )
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "timesnet":
        from model.timesnet.model import TimesNet
        # config_diff = config["diffusion"]
        model = TimesNet(
            data_name=args.data,
            num_features=config["data"]["num_features"],
            num_steps=config["data"]["num_steps"],
            num_layers=config["model"]["layers"],
            top_k=config["model"]["top_k"],
            d_model=config["model"]["dim_model"],
            d_ffn=config["model"]["dim_ffn"],
            num_kernels=config["model"]["num_kernels"],
            dropout=config["model"]['dropout'],
            device=args.device,
        )
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "mtsci":
        from model.mtsci.model import MTSCI
        config_diff = config["diffusion"]
        model = MTSCI(
            data_name=args.data,
            target_dim=config["data"]["num_features"],
            seqlen=config["data"]["num_steps"], # config_diff["seqlen"],
            layers=config_diff["layers"],
            nheads=config_diff["nheads"],
            channels=config_diff["channels"],
            num_steps=config_diff["num_steps"],
            timeemb=config["model"]["timeemb"],
            featureemb=config["model"]["featureemb"],
            diffusion_embedding_dim=config_diff["diffusion_embedding_dim"],
            schedule=config_diff["schedule"],
            beta_start=config_diff["beta_start"],
            beta_end=config_diff["beta_end"],
            alpha=config["train"]["alpha"],
            beta=config["train"]["beta"],
            # config, 
            device=args.device, 
            # seq_len=config["data"]["num_steps"]
        )
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "timemixer":
        from model.timemixer.model import TimeMixer
        # config_diff = config["diffusion"]
        model = TimeMixer(
            data_name=args.data,
            num_features=config["data"]["num_features"],
            num_steps=config["data"]["num_steps"],
            num_layers=config["model"]["layers"],
            top_k=config["model"]["top_k"],
            d_model=config["model"]["dim_model"],
            d_ffn=config["model"]["dim_ffn"],
            channel_independence=config["model"]["channel_independence"],
            decomp_method=config["model"]["decomp_method"],
            moving_avg=config["model"]["moving_avg"],
            downsampling_layers=config["model"]["down_sampling_layers"],
            downsampling_window=config["model"]["down_sampling_window"],
            dropout=config["model"]['dropout'],
            device=args.device,
        )
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "timemixerpp":
        from model.timemixerpp.model import TimeMixerPP
        # config_diff = config["diffusion"]
        model = TimeMixerPP(
            data_name=args.data,
            num_features=config["data"]["num_features"],
            num_steps=config["data"]["num_steps"],
            num_layers=config["model"]["layers"],
            top_k=config["model"]["top_k"],
            d_model=config["model"]["dim_model"],
            d_ffn=config["model"]["dim_ffn"],
            channel_independence=config["model"]["channel_independence"],
            decomp_method=config["model"]["decomp_method"],
            moving_avg=config["model"]["moving_avg"],
            downsampling_layers=config["model"]["down_sampling_layers"],
            downsampling_window=config["model"]["down_sampling_window"],
            dropout=config["model"]['dropout'],
            task_name=config["model"]['task_name'],
            n_heads=config["model"]['n_heads'],
            n_kernels=config["model"]['n_kernels'],
            channel_mixing=config["model"]['channel_mixing'],
            device=args.device,
        )
        optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    elif args.model == "ssdts":
        from model.ssdts.model import SSDTS

        config_diff = config["diffusion"]
        config_model = config["model"]
        model = SSDTS(
            data_name=args.data,
            num_features=config["data"]["num_features"],
            num_steps=config["data"]["num_steps"],
            layers=config_model["layers"],
            seq_dim=config_model["seq_dim"],
            res_channels=config_model["res_channels"],
            diffusion_embedding_dim=config_model["diffusion_embedding_dim"],
            diffusion_steps=config_diff["num_steps"],
            beta_start=config_diff["beta_start"],
            beta_end=config_diff["beta_end"],
            schedule=config_diff["schedule"],
            num_ssm=config_model["num_ssm"],
            cond_ssm_num=config_model["cond_ssm_num"],
            input_ssm_num=config_model["input_ssm_num"],
            num_ch=config_model["num_ch"],
            expand_c=config_model["expand_c"],
            expand_s=config_model["expand_s"],
            headdim_c=config_model["headdim_c"],
            headdim_s=config_model["headdim_s"],
            only_generate_missing=config_model.get("only_generate_missing", True),
            valid_all_steps=config_model.get("valid_all_steps", False),
            device=args.device,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=config["train"]["lr"],
            weight_decay=1e-6,
        )
    return model, optimizer
