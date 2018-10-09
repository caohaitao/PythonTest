__author__ = 'ck_ch'
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class LossShow():
    def __init__(self,loss_count,colors,labels):
        self.loss_count = loss_count
        self.index = 0
        self.xlabel = "x"
        self.ylabel = "y"
        self.title = "t"
        self.colors = colors
        self.labels = labels

    def set_xlabel(self,xlabel):
        self.xlabel = xlabel

    def set_ylabel(self,ylabel):
        self.ylabel = ylabel

    def set_title(self,title):
        self.title = title

    def __plt_init_internal__(self,ax):
        ax.axis("equal") #设置图像显示的时候XY轴比例
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title(self.title)
        plt.grid(True) #添加网格
        return ax

    def plt_init(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax = self.__plt_init_internal__(self.ax)
        plt.ion()
        self.xs = []
        self.ys = []

    def show(self,values):
        if len(values) != self.loss_count:
            return False

        plt.cla()
        self.ax = self.__plt_init_internal__(self.ax)
        for i in range(self.loss_count):
            if len(self.xs) != self.loss_count:
                self.xs.append([self.index])
            else:
                self.xs[i].append(self.index)
            if len(self.ys) != self.loss_count:
                self.ys.append([values[i]])
            else:
                self.ys[i].append(values[i])
            self.ax.plot(self.xs[i],self.ys[i],'-g',color=self.colors[i],label=self.labels[i])
            plt.legend(loc='best')


        self.index = self.index+1

        plt.pause(0.001)

    def stop(self):
        plt.ioff()
        plt.show()

# if __name__ == "__main__":
#     ls = LossShow(2,["red","blue"])
#     ls.plt_init()
#     for i in range(100):
#         values = [i,2*i]
#         ls.show(values)