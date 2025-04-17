# ftcsdi_ett_point_0.5_20250401_050254
# ftcsdi_ett_point_0.5_20250401_054949
# ftcsdi_ett_point_0.5_20250401_113230
# ftcsdi_ett_point_0.5_20250401_095642
# ftcsdi_ett_point_0.5_20250401_130308
# ftcsdi_ett_point_0.5_20250401_221802
# ftcsdi_ett_point_0.5_20250402_051641
# saits_ett_point_0.5_20250411_030512
import os
import pickle
import matplotlib.pyplot as plt


file = "saits_ett_point_0.5_20250411_030512"

model_ = file.split("_")[0]
data_ = file.split("_")[1]
time_ = file.split("_")[-1]

path = (f"./save/{file}/result_train_valid.pk")

# create figure
# 创建 2 行 2 列的子图
fig, axs = plt.subplots(2, 2, figsize=(20, 12))

def add_text(axs, epochs, infos, color):
    for x_val, y_val in zip(epochs, infos):
        axs.annotate(f'{y_val:.2f}',  # 显示保留两位小数
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 5),  # 相对于数据点偏移5个像素
                    ha='center',
                    fontsize=8,
                    color=color)


# first subplot
# load data file
with open(path, "rb") as f:
    observed_values = pickle.load(f)
for key in ["train", "valid"]:
    data = observed_values[key]
    epochs = [item['epoch'] for item in data]
    losses = [item['avg_epoch_loss'] for item in data]
    if key == "train":
        step = 10
        epochs = epochs[::step]
        losses = losses[::step]
        co = 'b'
    else:
        co = 'g'
    axs[1, 1].plot(epochs, losses, marker='o', linestyle='-', color=co, label=key)
    add_text(axs=axs[1, 1], epochs=epochs, infos=losses, color=co)

axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].set_title('Training and Validation Loss over Epochs')
axs[1, 1].grid(True)

# other three subplots
data = observed_values["test"]
# print(data)
RMSEs = [item["RMSE"].item() for item in data]
MAEs = [item["MAE"] for item in data]
MAPEs = [item["MAPE"] for item in data]
epochs = [i * 20 for i in range(len(data))]


# 保存图片到文件
# plt.savefig(f'ftcsdi_ett_2001_{key}ing_loss.png')

data = observed_values["test"]
# print(data)
RMSEs = [item["RMSE"].item() for item in data]
MAEs = [item["MAE"] for item in data]
MAPEs = [item["MAPE"] for item in data]
epochs = [i * 20 for i in range(len(data))]

axs[0, 0].plot(epochs, RMSEs, marker='o', linestyle='-', color='b')
add_text(axs=axs[0, 0], epochs=epochs, infos=RMSEs, color='b')
axs[0, 0].set_title('RMSE over Epochs')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('RMSE')
axs[0, 0].grid(True)

axs[0, 1].plot(epochs, MAEs, marker='o', linestyle='-', color='g')
add_text(axs=axs[0, 1], epochs=epochs, infos=MAEs, color='g')
axs[0, 1].set_title('MAE over Epochs')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('MAE')
axs[0, 1].grid(True)

axs[1, 0].plot(epochs, MAPEs, marker='o', linestyle='-', color='r')
add_text(axs=axs[1, 0], epochs=epochs, infos=MAPEs, color='r')
axs[1, 0].set_title('MAPE over Epochs')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('MAPE')
axs[1, 0].grid(True)

# 调整子图间距
plt.tight_layout()

# 保存图片到文件
plt.savefig(f'{model_}_{data_}_{time_}_results.png')

print(observed_values["save"])

path = (f"./save/{file}/result_nsample100.pk")

with open(path, "rb") as f:
    observed_values = pickle.load(f)
    print(observed_values)
