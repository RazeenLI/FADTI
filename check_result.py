import os
import pickle
import pandas as pd

foldname = "save"
root_dir = f'./{foldname}'  # 替换为你的save目录路径

records = []

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        result_path = os.path.join(folder_path, 'result_nsample100.pk')
        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                data = pickle.load(f)
            
            # 按名称提取分组字段，例如 csdi_ett_point_0.1_20250404_042120
            parts = folder.split('_')

            # 从后向前取出字段
            ratio = parts[-3]
            pattern = parts[-4]
            dataset = parts[-5]
            method = '_'.join(parts[:-5])  # 剩下的部分全是method
            # 把数据打包为一行，添加分组信息
            records.append({
                'method': method,
                'dataset': dataset,
                'pattern': pattern,
                'ratio': ratio,
                'rmse': data[0],
                'mae': data[1],
                'mape': data[2],
                'crps': data[3],
                'time': data[4],
            })


# 将记录转换为 DataFrame
df = pd.DataFrame(records)
df.to_csv(f'info/{foldname}_original_results.csv', index=False)

# 根据分组字段进行聚合（sum 和 mean）
group_cols = ['method', 'dataset', 'pattern', 'ratio']
sum_df = df.groupby(group_cols).sum(numeric_only=True).reset_index()
mean_df = df.groupby(group_cols).mean(numeric_only=True).reset_index()

# 统计每个分组的数量（子类数量）
count_df = df.groupby(group_cols).size().reset_index(name='count')
# 合并 count 信息到 mean_df
mean_df = mean_df.merge(count_df, on=group_cols)

# 需要保留三位小数的列
cols_to_round = ['mae', 'rmse', 'mape', 'crps', 'time']
# 保留三位小数
mean_df[cols_to_round] = mean_df[cols_to_round].round(3)

print(mean_df.head)
mean_df.to_csv(f'info/{foldname}_mean_results.csv', index=False)
# # 显示结果
# import ace_tools as tools; tools.display_dataframe_to_user(name="Sum of Results by Group", dataframe=sum_df)
# tools.display_dataframe_to_user(name="Mean of Results by Group", dataframe=mean_df)
