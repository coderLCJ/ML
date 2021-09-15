# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         jiebaTest
# Description:  
# Author:       Laity
# Date:         2021/9/12
# ---------------------------------------------
import jieba

content = '我每天晚上都要去操场跑步'
res = jieba.lcut(content, cut_all=True)
print(res)
