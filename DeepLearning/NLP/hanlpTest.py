# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         hanlpTest
# Description:  
# Author:       Laity
# Date:         2021/9/13
# ---------------------------------------------
import hanlp
# 加载英文命名实体识别的预训练模型PTB_POS_RNN_FASTTEXT_EN
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
# 输入是分词结果列表
print(tagger)
