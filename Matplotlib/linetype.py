# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y1)


plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
#设置x轴的范围
plt.xlim((-2,2))
#设置x轴的标签
plt.xlabel('i am x')
plt.ylim((-3,3))
plt.ylabel('i am y')
#生成刻度范围，-2-2,5个刻度
new_ticks = np.linspace(-2,2,5)
#设置x轴的刻度
plt.xticks(new_ticks)
plt.show()