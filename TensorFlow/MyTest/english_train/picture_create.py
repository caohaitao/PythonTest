# -*- coding: utf-8 -*-
__author__ = 'ck_ch'
import cv2
import numpy as np

def create_picture(c):
    width = 28
    height = 28
    image = np.full((height,width,3),fill_value=255,dtype=np.uint8)
    font=cv2.FONT_HERSHEY_TRIPLEX
    #照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
    cv2.putText(image,c,(4,height-4),font,1,(0,0,255),1,False)
    file_name='data\\'+c+'.jpg'
    print(file_name)
    cv2.imwrite(file_name,image)



for i in range(26):
    create_picture(chr(i+ord('A')))



# cv2.imshow('img',image)
# cv2.waitKey(0)