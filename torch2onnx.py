 # 构造模型实例
from architecture import *
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt

model = model_generator(opt.method, r'./simulation/train_code/exp/restormer_nonoise_2/2023_08_14_17_23_07/model/model_epoch_44.pth')

# 反序列化权重参数
model.eval()
# 定义输入名称，list结构，可能有多个输入
input_names = ['input']
# 定义输出名称，list结构，可能有多个输出
output_names = ['output']
# 构造输入用以验证onnx模型的正确性
input = torch.rand(1, 1, 1024, 1024)
# 导出
torch.onnx.export(model, input, './simulation/train_code/tronnx1_best.onnx',
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names)