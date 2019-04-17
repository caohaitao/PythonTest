__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import sys

def show_file_loss(file_path):
    y = []

    for line in open(file_path):
        y.append(float(line))

    plt.plot(y)
    plt.show()


def fwrite_loss(file_path,loss):
    f = open(file_path,'a+')
    s = format("%0.6f\n"%loss)
    f.writelines(s)
    f.close()

if __name__=='__main__':
    show_file_loss(sys.argv[1])