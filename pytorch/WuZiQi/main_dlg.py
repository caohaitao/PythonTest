from tkinter import *
import tkinter.messagebox
from Board import *

BOARD_LENGTH = 500

top = Tk()
top.geometry("600x600")

def mouse_click(event):
    print("click x=%d,y=%d"%(event.x,event.y))

def GetXByPos(i,x_begin,one_bar_length):
    return x_begin + i*one_bar_length

def GetYByPos(j,y_begin,one_bar_length):
    return int(y_begin + j*one_bar_length)

def hit_ok():
    #lambda x=ALL:canvas.delete(x)
    x_begin = 10
    y_begin = 10
    for x in canvas.find_all():
        canvas.delete(x)
    canvas.create_rectangle(0,0,BOARD_LENGTH,BOARD_LENGTH,fill="white")
    one_bar_length = int(BOARD_LENGTH / (BOARD_SIZE-1))
    chess_radius = int(one_bar_length / 4)
    for i in range(BOARD_SIZE):
        y_pos = GetYByPos(i,y_begin,one_bar_length)
        canvas.create_line(x_begin,y_pos,x_begin+)

canvas = Canvas(top,width=BOARD_LENGTH,height=BOARD_LENGTH)
canvas.bind("<Button-1>",mouse_click)
canvas.pack()
#l = canvas.create_line(0,0,100,100,fill="red",dash=(4,4))

B1 = Button(top,text="b1",command=hit_ok)
B1.pack()
top.mainloop()
