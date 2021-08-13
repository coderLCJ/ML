import pandas as pd
import torch

data = pd.read_csv('data/house_tiny.csv')
print(data)

# 处理缺失值
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]

inputs = inputs.fillna(inputs.mean())   # 用均值替换缺失值
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为张量
x = torch.tensor(inputs.values)
y = torch.tensor(outputs.values)
print(x, '\n', y)
