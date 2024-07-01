import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
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
import cv2


batch_size_all=1 

engine_path = "./simulation/train_code/tronnx1_best.engine"
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open(engine_path, "rb") as f:
    engine_data = f.read()

engine = runtime.deserialize_cuda_engine(engine_data)

_, test_data_low = LoadTest(f"{opt.data_root}")  # 已修改训练集

context = engine.create_execution_context()
input_shape = (batch_size_all, 1, 1024, 1024)  # 输入的形状
output_shape = (batch_size_all, 1, 1024, 1024)  # 输出的形状

# 分配输入和输出内存
input_host = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
output_host = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
input_device = cuda.mem_alloc(input_host.nbytes)
output_device = cuda.mem_alloc(output_host.nbytes)
bindings = [int(input_device), int(output_device)]
stream = cuda.Stream()

psnr_list=[]
time_list=[]
filename1='./simulation/train_code/psnr1.txt'
filename2='./simulation/train_code/time1.txt'

for i in range(len(test_data_low) // batch_size_all):
    test_gt, input_meas = shuffle_crop4(test_data_low, test_data_low, batch_size_all, crop_size=1024, pianyi = i)
    # 准备输入数据
    input_data = input_meas
    # 输入数据
    np.copyto(input_host, input_data.cpu().ravel())
    # 将输入数据从主机内存复制到设备内存
    cuda.memcpy_htod_async(input_device, input_host, stream)
    t1=time.time()
    # 进行推理
    context.execute_async(batch_size=batch_size_all, bindings=bindings, stream_handle=stream.handle)
    t2=time.time()
    # 将输出数据从设备内存复制到主机内存
    cuda.memcpy_dtoh_async(output_host, output_device,stream)
    # 等待异步推理完成
    stream.synchronize()
    # 获取输出结果
    result = np.array(output_host).reshape(output_shape)
    result=np.array(result)
    print('result:',result.shape)
    for k in range(batch_size_all):
        result1=result[k,:,:,:]
        # print('result1',result1) 
        # print('result1.shape:',result1.shape)
        # result1= np.squeeze(result1) 
        # cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Gray Image", 256, 256)
        # cv2.imshow("Gray Image", result1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    
        print('time',t2-t1) 
        t=t2-t1
        test_gt = torch.tensor(test_gt)
        result1 = torch.tensor(result1) 
        psnr_val = torch_psnr(result1, test_gt[k, :, :, :])
        psnr_list.append(psnr_val.detach().cpu().numpy())
        time_list.append(t)
        with open(filename1,'w') as file:
            for item in psnr_list:
                file.write(str(item)+'\n')

        with open(filename2,'w') as file:
            for item in time_list:
                file.write(str(item)+'\n')
                
        
        