from OpenGL.GL import *
from OpenGL.GLUT import *


def draw():
    # ---------------------------------------------------------------
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-0.8, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(0.8, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）

    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.0, -0.8, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, 0.8, 0.0)  # 设置y轴顶点（y轴正方向）

    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.0, -0.8)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, 0.8)  # 设置z轴顶点（z轴正方向）

    glEnd()  # 结束绘制线段

    # ---------------------------------------------------------------
    glBegin(GL_TRIANGLES)  # 开始绘制三角形（z轴负半区）

    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glVertex3f(-0.5, -0.366, -0.5)  # 设置三角形顶点
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glVertex3f(0.5, -0.366, -0.5)  # 设置三角形顶点
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glVertex3f(0.0, 0.5, -0.5)  # 设置三角形顶点

    glEnd()  # 结束绘制三角形

    # ---------------------------------------------------------------
    glFlush()  # 清空缓冲区，将指令送往硬件立即执行


if __name__ == "__main__":
    glutInit()  # 1. 初始化glut库
    glutCreateWindow(b'Quidam Of OpenGL')  # 2. 创建glut窗口

    glClearColor(1.0,1.0,1.0,0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    l = 2
    glOrtho(-l,l,-l,l,-100.0,100.0)

    glutDisplayFunc(draw)  # 3. 注册回调函数draw()
    glutMainLoop()  # 4. 进入glut主循环
