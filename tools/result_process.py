'''
处理训练生成的批量误差数据，目前支持同一文件内有且只有一种模型的数据
Usage: 修改 model_name
'''

import matplotlib.pyplot as plt
import pandas as pd
import io
import re
import numpy as np
from sympy import false

model_name = 'Crossformer'

def extract_model_data(data, model_name):
    model_data = []
    count = false
    for line in data:
        if count:
            model_data.append(line)
            count = not count
        match = re.search(model_name, line)
        if match:
            model_data.append(line)
            count = not count
    return model_data


with open('../result_long_term_forecast.txt', 'r', encoding='utf-8') as file:
    data_all = file.read()

data_lines = data_all.strip().split('\n')
raw_data = extract_model_data(data_lines, model_name)

# 设置 pandas 显示选项以显示完整的 DataFrame
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽

# 将字符串数据转换为 pandas DataFrame
processed_data = []
i = 0
while i < len(raw_data):
    if raw_data[i].strip():  # 跳过空行
        # 合并实验名称和指标数据（假设实验名称和指标数据在相邻的两行）
        if i + 1 < len(raw_data) and raw_data[i + 1].strip().startswith('mse:'):
            experiment_name = raw_data[i].strip()
            metrics = raw_data[i + 1].strip()
            i += 2
        else:
            # 如果指标数据不在下一行，则尝试从当前行提取
            print("指标数据不在下一行")
            parts = raw_data[i].split('  ')
            if len(parts) == 2:
                experiment_name = parts[0].strip()
                metrics = parts[1].strip()
                i += 1
            else:
                print(f"Warning: Unexpected line format - {raw_data[i]}")
                i += 1
                continue

        # 提取 mse 和 mae 的值
        mse_match = re.search(r'mse:([^,]+)', metrics)
        mae_match = re.search(r'mae:([^,]+)', metrics)

        if mse_match and mae_match:
            mse_val = mse_match.group(1)
            mae_val = mae_match.group(1)
            processed_data.append({
                'exp_name': experiment_name,
                'mse': float(mse_val),
                'mae': float(mae_val)
            })
        else:
            print(f"Warning: Could not extract mse or mae from line - {metrics}")
    else:
        i += 1

print(processed_data)
# 创建 DataFrame
df = pd.DataFrame(processed_data)

df[['dataset', 'pred_lens', 't_kernel_sizes', 'v_kernel_sizes', 'model_name', '_', 'features', '_']] = df['exp_name'].str.extract(
    r'long_term_forecast_([a-zA-Z0-9]+)_96_([\d.]+)_([\d.]+)_([\d.]+)_([a-zA-Z0-9]+_dCNN)_([a-zA-Z0-9]+)_ft([a-zA-Z]+)_([a-zA-Z0-9_])')

# 转换参数为数值类型
df['pred_lens'] = df['pred_lens'].astype(float)
df['t_kernel_sizes'] = df['t_kernel_sizes'].astype(float)
df['v_kernel_sizes'] = df['v_kernel_sizes'].astype(float)

print(df)

# 按数据集名称分组并绘制图表
for ds in df['dataset'].unique():
    df_ds = df[df['dataset'] == ds]

    # 找到最小的 MSE 和 MAE 值及其对应的参数
    min_mse_row = df_ds.loc[df_ds['mse'].idxmin()]
    min_mae_row = df_ds.loc[df_ds['mae'].idxmin()]

    # 计算平均 MSE 和 MAE
    avg_mse = df_ds['mse'].mean()
    avg_mae = df_ds['mae'].mean()

    # 打印结果
    print(f"Dataset: {ds}")
    print(
        f"Minimum MSE: {min_mse_row['mse']:.4f} (t_kernel_sizes: {min_mse_row['t_kernel_sizes']}, v_kernel_sizes: {min_mse_row['v_kernel_sizes']}, pred_lens: {min_mse_row['pred_lens']})")
    print(
        f"Minimum MAE: {min_mae_row['mae']:.4f} (t_kernel_sizes: {min_mae_row['t_kernel_sizes']}, v_kernel_sizes: {min_mae_row['v_kernel_sizes']}, pred_lens: {min_mae_row['pred_lens']})")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print("-" * 50)

    pred_lens = df_ds['pred_lens'].unique()

    num_rows = len(pred_lens)
    if num_rows == 0:
        continue

    # 创建一个独立的图表用于当前数据集
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    # 如果只有一个预测长度，调整 axes 的形状
    if num_rows == 1:
        axes = axes.reshape(1, 2)

    for idx, pred_len in enumerate(pred_lens):
        grouped_df = df_ds[df_ds['pred_lens'] == pred_len]

        # 创建透视表
        pivot_table_mse = grouped_df.pivot_table(index='t_kernel_sizes', columns='v_kernel_sizes', values='mse',
                                                 aggfunc='mean')
        pivot_table_mae = grouped_df.pivot_table(index='t_kernel_sizes', columns='v_kernel_sizes', values='mae',
                                                 aggfunc='mean')

        # 绘制 MSE 热图
        ax_mse = axes[idx, 0]
        im_mse = ax_mse.imshow(pivot_table_mse, cmap='viridis', aspect='auto')
        fig.colorbar(im_mse, ax=ax_mse, label='MSE')
        ax_mse.set_title(f'pred_lens = {pred_len} (MSE)')
        ax_mse.set_xlabel('v_kernel_sizes')
        ax_mse.set_ylabel('t_kernel_sizes')
        ax_mse.set_xticks(np.arange(len(pivot_table_mse.columns)))
        ax_mse.set_xticklabels(pivot_table_mse.columns)
        ax_mse.set_yticks(np.arange(len(pivot_table_mse.index)))
        ax_mse.set_yticklabels(pivot_table_mse.index)
        for i in range(len(pivot_table_mse.index)):
            for j in range(len(pivot_table_mse.columns)):
                mse_val = pivot_table_mse.iloc[i, j]
                ax_mse.text(j, i, f"{mse_val:.4f}", ha="center", va="center", color="w", fontsize=8)

        # 绘制 MAE 热图
        ax_mae = axes[idx, 1]
        im_mae = ax_mae.imshow(pivot_table_mae, cmap='viridis', aspect='auto')
        fig.colorbar(im_mae, ax=ax_mae, label='MAE')
        ax_mae.set_title(f'pred_lens = {pred_len} (MAE)')
        ax_mae.set_xlabel('v_kernel_sizes')
        ax_mae.set_ylabel('t_kernel_sizes')
        ax_mae.set_xticks(np.arange(len(pivot_table_mae.columns)))
        ax_mae.set_xticklabels(pivot_table_mae.columns)
        ax_mae.set_yticks(np.arange(len(pivot_table_mae.index)))
        ax_mae.set_yticklabels(pivot_table_mae.index)
        for i in range(len(pivot_table_mae.index)):
            for j in range(len(pivot_table_mae.columns)):
                mae_val = pivot_table_mae.iloc[i, j]
                ax_mae.text(j, i, f"{mae_val:.4f}", ha="center", va="center", color="w", fontsize=8)

    plt.tight_layout()
    plt.suptitle(f'Dataset: {ds}', y=0.99, fontsize=10)  # 添加数据集标题
    plt.show()