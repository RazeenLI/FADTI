import torch

SUPPORTED_DATASETS = {"ett", "weather", "metr_la", "ecoli"}


def get_process_data(data_name):
    if data_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {data_name}")
    return process_data_standard


def process_data_standard(batch, device):
    observed_data = batch["observed_data"].to(device).float()
    observed_mask = batch["observed_mask"].to(device).float()
    observed_tp = batch["timepoints"].to(device).float()
    next_data = batch["next_data"].to(device).float()
    gt_mask = batch["gt_mask"].to(device).float()

    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    gt_mask = gt_mask.permute(0, 2, 1)
    next_data = next_data.permute(0, 2, 1)

    cut_length = torch.zeros(len(observed_data)).long().to(device)
    for_pattern_mask = observed_mask

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "observed_tp": observed_tp,
        "gt_mask": gt_mask,
        "next_data": next_data,
        "for_pattern_mask": for_pattern_mask,
        "cut_length": cut_length,
    }
