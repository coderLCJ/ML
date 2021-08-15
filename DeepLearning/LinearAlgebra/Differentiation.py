import matplotlib
import numpy as np
from IPython import display

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h


h = 0.1
for i in range(5):
    print(f'h={h:.5f}, limit={numerical_lim(f, 1, h):.5f}') # f开头表示在字符串内支持大括号内的python 表达式，
    h *= 0.1