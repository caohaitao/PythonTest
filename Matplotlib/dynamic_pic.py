__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import time
import math

def Method_Improve(point):
    def initial(ax):
        ax.axis("equal") #设置图像显示的时候XY轴比例
        ax.set_xlabel('Horizontal Position')
        ax.set_ylabel('Vertical Position')
        ax.set_title('Vessel trajectory')
        plt.grid(True) #添加网格
        return ax

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax = initial(ax)
    plt.ion()
    print('开始仿真')
    obsX = [0,]
    obsY = [0,]
    for t in point:
        obsX.append(t[0])
        obsY.append(t[1])
        plt.cla()
        ax = initial(ax)
        ax.plot(obsX,obsY,'-g',marker='*',color="red",label="cht")
        plt.legend(loc='best')
        plt.pause(0.01)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    points = []
    for i in range(100):
        points.append([i,100-i])
    Method_Improve(points)


