import torch
import random

# def synthetic_data(w, b, num_examples):
#     X = torch.normal(0, 1, size=(num_examples, len(w)))
#     y = torch.matmul(X, w) + b
#     y += torch.normal(0, 0.001, size=y.shape)
#     return X, y.reshape((-1, 1))
#
#
# true_w = torch.tensor([2, -3.4])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 10)
# print(features, labels)

x = torch.normal(mean=0.5, std=torch.arange(1, 6))
print(x)