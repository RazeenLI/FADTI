import torch

def get_process_data(dataname):
    if dataname == "physio":
         return process_data_physio
    elif dataname == "metrla":
         return process_data_metrla
    else:
         return process_data_physio

def process_data_physio(batch, device):
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

    
def process_data_metrla(batch, device):
    observed_data = batch["observed_data"].to(device).float()
    observed_mask = batch["observed_mask"].to(device).float()
    observed_tp = batch["timepoints"].to(device).float()
    next_data = batch["next_data"].to(device).float()
    gt_mask = batch["gt_mask"].to(device).float()
    cut_length = batch["cut_length"].to(device).long()
    # coeffs = None
    # if self.config['model']['use_guide']:
    #     coeffs = batch["coeffs"].to(device).float()
    # cond_mask = batch["cond_mask"].to(device).float()

    observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
    observed_mask = observed_mask.permute(0, 2, 1)
    gt_mask = gt_mask.permute(0, 2, 1)
    next_data = next_data.permute(0, 2, 1)
    # cond_mask = cond_mask.permute(0, 2, 1)
    for_pattern_mask = observed_mask

    # if self.config['model']['use_guide']:
    #     coeffs = coeffs.permute(0, 2, 1)

    return {
        "observed_data": observed_data,
        "observed_mask": observed_mask,
        "observed_tp": observed_tp,
        "gt_mask": gt_mask,
        "for_pattern_mask": for_pattern_mask,
        "cut_length": cut_length,
        "next_data": next_data,
        # "coeffs": coeffs,
        # "cond_mask": cond_mask,
    }