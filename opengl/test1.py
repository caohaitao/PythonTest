from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from threading import Timer
import time
from stl_reader import read_one_file

triangles = []

def drawFunc():
    global triangles
    # 清楚之前画面
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    # glRotatef(0.1, 5, 5, 0)  # (角度,x,y,z)
    # glutWireTeapot(0.5)

    glColor3f(1.0,0.0,0.0)
    glPushMatrix()
    glBegin(GL_TRIANGLES)
    for t in triangles:
        print(t.normal,t.pnts)
        glNormal3f(t.normal[0],t.normal[1],t.normal[2])
        for i in range(3):
            glVertex3f(t.pnts[i][0],t.pnts[i][1],t.pnts[i][2])
    glEnd()

    glPopMatrix()
    # 刷新显示
    glFlush()
    glutSwapBuffers()

def f1():
    glutPostRedisplay()
    Timer(0.01,f1).start()

def mouseButton(button,mode,x,y):

    which = 'right'
    m = 'down'

    if button == GLUT_RIGHT_BUTTON:
        which = 'right'
    else:
        which = 'left'
    if mode == GLUT_DOWN:
        m = 'down'
    else:
        m = 'up'

    print(which,m,x,y)

if __name__=='__main__':
    triangles = read_one_file(r"D:\code\MyCodes\CHT3D\modelmatch\stlModels\mytest\rect0.stl")

    # 使用glut初始化OpenGL
    b = glutInit()
    if b == False:
        print("glutInit failed")
        exit(0)
    # 显示模式:GLUT_SINGLE无缓冲直接显示|GLUT_RGBA采用RGB(A非alpha)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)

    # 窗口位置及大小-生成
    glutInitWindowPosition(0, 0)
    glutInitWindowSize(400, 400)
    glutCreateWindow(b"first")
    # 调用函数绘制图像
    glutDisplayFunc(drawFunc)
    # glutIdleFunc(drawFunc)
    glutMouseFunc(mouseButton)
    # Timer(0.1,f1).start()
    # 主循环
    glutMainLoop()