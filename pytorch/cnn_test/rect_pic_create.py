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

def create_a_pic(r):

    img = Image.new('RGB',(width,height),(255,255,255))
    draw = ImageDraw.Draw(img)
    one_len=int(width/one_row_nums)
    x = int(int(r)%int(one_row_nums))
    y = int(int(r)/int(one_row_nums))
    draw.rectangle(((x*one_len, y*one_len), (x*one_len+one_len, y*one_len+one_len)), fill="black")
    file_path = format("data\\rect_%d.jpg"%(r))
    img.save(file_path,'jpeg')

if __name__ == "__main__":
    if not os.path.exists("data\\"):
        os.mkdir("data\\")
    for i in range(one_row_nums*one_row_nums):
        create_a_pic(i)
