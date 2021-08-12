import pandas as pd


data = pd.read_csv('data/house_tiny.csv')
print(data)

# 处理缺失值
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)