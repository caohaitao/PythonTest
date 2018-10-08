__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import io
import sys
import os
from PIL import Image,ImageDraw,ImageFont
import random

width = 32
height = 32
one_row_nums = 8

color_maps = [[255,0,0],[0,255,0],[0,0,255]]

def create_a_pic(folder_path,r):

    img = Image.new('RGB',(width,height),(255,255,255))
    draw = ImageDraw.Draw(img)
    # one_len=int(width/one_row_nums)
    # x = int(int(r)%int(one_row_nums))
    # y = int(int(r)/int(one_row_nums))
    # color_index = random.randint(0,2)
    # draw.rectangle(((x*one_len, y*one_len), (x*one_len+one_len, y*one_len+one_len)), fill=(color_maps[color_index][0],color_maps[color_index][1],color_maps[color_index][2]))
    # file_path = format("%s\\rect_%d_%d.jpg"%(folder_path,color_index,r))
    # img.save(file_path,'jpeg')

    one_len = 4
    x = random.randint(0,width-1)
    y = random.randint(0,height-1)
    color_index = random.randint(0,2)
    draw.rectangle(((x-one_len/2, y-one_len/2), (x+one_len/2, y+one_len/2)), fill=(color_maps[color_index][0],color_maps[color_index][1],color_maps[color_index][2]))
    file_path = format("%s\\rect_%d_%d_%d.jpg"%(folder_path,color_index,x,y))
    img.save(file_path,'jpeg')

if __name__ == "__main__":
    if not os.path.exists("data\\"):
        os.mkdir("data\\")
    # for i in range(one_row_nums*one_row_nums):
    #     create_a_pic("data",i)
    for i in range(1000):
        a = random.randint(0,one_row_nums*one_row_nums-1)
        create_a_pic("data",a)

    if not os.path.exists("test\\"):
        os.mkdir("test\\")
    for i in range(one_row_nums*one_row_nums):
        create_a_pic("test",i)
