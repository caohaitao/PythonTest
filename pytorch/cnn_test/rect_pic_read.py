__author__ = 'ck_ch'
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision

def read_one_data(file_path):
    im = cv2.imread(file_path)
    w = im.shape[0]
    h = im.shape[1]
    data = (cv2.cvtColor(im,cv2.COLOR_BGR2RGB)/255.0).reshape(3,w,h).astype(np.float32)
    return data,w,h
    # cv2.imshow("image",gray)
    # cv2.waitKey(0)

def read_one_data2(file_path):
    image = Image.open(file_path)
    image = np.array(image,dtype=np.float32)
    img = torch.from_numpy(image)
    w = image.shape[0]
    h = image.shape[1]
    c = image.shape[2]
    img = img.view(w,h,c)
    img = img.transpose(0,1).transpose(0,2).contiguous()
    return img,w,h

def read_datas(dir):
    #dir = 'data\\'
    for (root,dirs,files) in os.walk(dir):
        print(len(files))
        #res = np.ndarray(shape=(len(files),1,width,height),dtype='float32')
        label = np.ndarray(shape=(len(files)),dtype='int64')
        label2 = np.ndarray(shape=(len(files),2),dtype='float32')
        i = 0
        for item in files:
            file_path = format("%s%s"%(dir,item))
            r,w,h = read_one_data2(file_path)
            if i==0:
                res = np.ndarray(shape=(len(files),3,w,h),dtype='float32')
            res[i]= r
            sl = item.replace('.jpg','')
            sls = sl.split('_')
            label[i] = int(sls[1])
            label2[i][0] = float(sls[2])
            label2[i][1] = float(sls[3])
            i = i+1
    return res,label,label2,w,h



# if __name__ == "__main__":
#     #read_datas()
#     read_one_data2()