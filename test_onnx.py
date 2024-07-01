import torch
import numpy
import onnxruntime
import os
import numpy as np
from torch.autograd import Variable
import datetime
from option import opt
import cv2
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
_, test_data_low = LoadTest(f"{opt.data_root}")
print('array:',np.array(test_data_low).shape)
os.environ["OMP_NUM_THREADS"] = "4"
def main():
    for i in range(len(test_data_low) // 1):
    test_gt, input_meas = shuffle_crop4(test_data_low, test_data_low, 1, crop_size=128, pianyi = i)
    
    dummy_input = input_meas
    session = onnxruntime.InferenceSession("/home/nvidia/Downloads/gt10_27/simulation/train_code/tronnx1.onnx")
    result = session.run([], {"input":dummy_input.numpy()})
    result=np.array(result)
    result = np.squeeze(result)
    cv2.imshow("Gray Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('array:',result.shape)