__author__ = 'ck_ch'
from PIL import Image
import sys
import numpy as np

def rgb_to_gray(pixel):
    return int(0.39*pixel[0]+0.5*pixel[1]+0.11*pixel[2])

def read_one_bbmp(bmp_path):
    im = Image.open(bmp_path)
    res = np.ndarray(shape=(784),dtype=int)
    width = im.size[0]
    height = im.size[1]
    print("width=%d,height=%d" % (width,height))
    count=0
    for h in range(0,height):
        for w in range(0,width):
            pixel = im.getpixel((w,h))
            gray_pixel = rgb_to_gray(pixel)
            if gray_pixel > 150:
                res[count] = 0
            else:
                res[count] = 1
            count+=1
    return res


b1 = read_one_bbmp("test1.bmp")

i=0
for a in b1:
    if i%28 == 0:
        sys.stdout.write('\n')
    if a== 0:
        #print(0),
        sys.stdout.write(str(0))
    else:
        sys.stdout.write(str(1))
    sys.stdout.write(" ")
    i=i+1
sys.stdout.write('\n')
