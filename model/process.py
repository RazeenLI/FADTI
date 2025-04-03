import numpy as np
import torch
from torch.optim import AdamW, SGD
from tqdm import tqdm
from typing import Union, Optional
import pickle
# from nn.looksam import LookSAM
from nn.sam import SAM

class Process(object):
    def __init__(
            self,
            model,
            learning_rate,
            device: Optional[Union[str, torch.device, list]] = None,
            epochs: int = 100,
            batch_size: int = 32,
            patience: Optional[int] = None,
            save_strategy="new" # "new", "best", "none", "all"
        ):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.saving_path = None
        self.save_strategy = save_strategy

        # set up model
        self.model = model.to(self.device)
        
        # set up optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        # self.optimizer = SGD(self.model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-6)
        # base_optimizer = AdamW
        # self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=learning_rate, weight_decay=1e-6)
        # self.optimizer = LookSAM(self.model.parameters(), base_optimizer, lr=learning_rate, rho=0.05, alpha=0.5, k=5)
        # p1 = int(0.75 * epochs)
        # p2 = int(0.9 * epochs)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)


        self.info = {
            "train": [],
            "valid": [],
            "test": []
        }
    
    def train(
            self,
            # config,
            train_loader,
            valid_loader=None,
            test_loader=None,
            test_epoch_interval=20,
            valid_epoch_interval=10,
            foldername="",
    ):
        if foldername != "":
            output_path = foldername + "/model.pth"

        best_valid_loss = 1e10
        for epoch in range(self.epochs):
            avg_loss = 0
            # epoch_train_loss_collector = []
            self.model.train()
            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_index, batch_data in enumerate(it, start=1):
                    def closure():
                        # it.write(f"Inside closure for batch {batch_index}")
                        self.optimizer.zero_grad()    # 每次调用 closure 前清零梯度
                        loss = self.model(batch_data)
                        loss.backward()
                        # 调试输出：打印梯度非空参数的数量
                        # grad_count = sum(p.grad is not None for p in self.model.parameters())
                        # it.write(f"Batch {batch_index}: Number of parameters with gradients: {grad_count}")
                        return loss
                    # 使用 SAM/LookSAM 的 step()，它会内部调用 closure 两次
                    loss = closure()
                    # self.optimizer.step(closure)
                    # avg_loss += loss.item()

                    # self.optimizer.zero_grad()
                    # loss = self.model(batch_data)
                    # loss.backward()

                    avg_loss += loss.item()
                    self.optimizer.step()

                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_loss / batch_index,
                            "batch_index": batch_index,
                            "epoch": epoch,
                        },
                        refresh=False,
                    )
                    # if batch_index >= config["itr_per_epoch"]:
                    #     break
                self.info["train"].append({
                    "avg_epoch_loss": avg_loss / batch_index,
                    "epoch": epoch,
                })

                self.scheduler.step()
            if valid_loader is not None and (epoch + 1) % valid_epoch_interval == 0:
                self.model.eval()
                avg_loss_valid = 0
                with torch.no_grad():
                    with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                        for batch_index, valid_batch in enumerate(it, start=1):
                            loss = self.model(valid_batch)
                            avg_loss_valid += loss.item()
                            # val_loss_collector.append(loss.item())
                            it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": avg_loss_valid / batch_index,
                                    "epoch": epoch,
                                },
                                refresh=False,
                            )
                        self.info["valid"].append({
                            "avg_epoch_loss": avg_loss_valid / batch_index,
                            "epoch": epoch,
                        })
                best_valid_loss = self.save_model(output_path, avg_loss_valid / batch_index, best_valid_loss, epoch)

            if test_loader is not None and (epoch + 1) % test_epoch_interval == 0:
                self.evaluate(
                    test_loader=test_loader, 
                    scaler=1, 
                    foldername=foldername
                )

        if self.save_strategy == "new":
            best_valid_loss = self.save_model(output_path, best_valid_loss, best_valid_loss, self.epochs)
            torch.save(self.model.state_dict(), output_path)
        
        with open(
            foldername + "/result_train_valid.pk", "wb"
        ) as f:
            pickle.dump(self.info, f,)
    

    def save_model(self, save_path, current_valid_loss, best_valid_loss, epoch):
        # 根据保存策略判断是否保存模型
        if self.save_strategy == "best":
            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                print("\nBest loss updated to", best_valid_loss, "at epoch", epoch)
                torch.save(self.model.state_dict(), save_path)
        elif self.save_strategy == "new":
            print("\nSaving new model at epoch", epoch)
            torch.save(self.model.state_dict(), save_path)
        elif self.save_strategy == "all":
            # 每个 epoch 保存一个不同的文件，可以在文件名中加入 epoch 信息
            epoch_output_path = save_path + f"_epoch_{epoch}"
            print("\nSaving model for epoch", epoch)
            torch.save(self.model.state_dict(), epoch_output_path)
        else:
            # 什么也不保存
            return best_valid_loss
        self.info["save"] = {
            "save strategy": self.save_strategy,
            "epoch": epoch
        }
        return best_valid_loss


    def evaluate(
            self,
            test_loader, 
            nsample=100, 
            scaler=1, 
            mean_scaler=0, 
            foldername=""
    ):
        with torch.no_grad():
            self.model.eval()
            mse_total = 0
            mae_total = 0
            mape_total = 0
            evalpoints_total = 0

            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_index, batch_data in enumerate(it, start=1):
                    output = self.model.evaluate(batch_data, nsample)

                    samples, c_target, eval_points, observed_points, observed_time = output
                    
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    samples_median = samples.median(dim=1)
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    error = (samples_median.values - c_target) * eval_points
                    error_scaled = error * scaler

                    mse_current = (error_scaled ** 2)
                    mae_current = torch.abs(error_scaled)
                    mape_current = torch.abs(error_scaled / (c_target * scaler + 1e-8)) * eval_points

                    # mse_current = (
                    #     ((samples_median.values - c_target) * eval_points) ** 2
                    # ) * (scaler ** 2)
                    # mae_current = (
                    #     torch.abs((samples_median.values - c_target) * eval_points) 
                    # ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    mape_total += mape_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "mape_current": mape_total / evalpoints_total,
                            "batch_no": batch_index,
                        },
                        refresh=True,
                    )

                with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    all_target = torch.cat(all_target, dim=0)
                    all_evalpoint = torch.cat(all_evalpoint, dim=0)
                    all_observed_point = torch.cat(all_observed_point, dim=0)
                    all_observed_time = torch.cat(all_observed_time, dim=0)
                    all_generated_samples = torch.cat(all_generated_samples, dim=0)

                    pickle.dump(
                        [
                            all_generated_samples,
                            all_target,
                            all_evalpoint,
                            all_observed_point,
                            all_observed_time,
                            scaler,
                            mean_scaler,
                        ],
                        f,
                    )

                CRPS = calc_quantile_CRPS(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )
                CRPS_sum = calc_quantile_CRPS_sum(
                    all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
                )

                with open(
                    foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
                ) as f:
                    pickle.dump(
                        [
                            np.sqrt(mse_total / evalpoints_total),
                            mae_total / evalpoints_total,
                            mape_total / evalpoints_total,
                            CRPS,
                        ],
                        f,
                    )
                    print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                    print("MAE:", mae_total / evalpoints_total)
                    print("MAPE:", mape_total / evalpoints_total)
                    print("CRPS:", CRPS)
                    print("CRPS_sum:", CRPS_sum)
                self.info["test"].append({
                    "RMSE": np.sqrt(mse_total / evalpoints_total),
                    "MAE": mae_total / evalpoints_total,
                    "MAPE": mape_total / evalpoints_total
                })

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)
