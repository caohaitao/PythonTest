from tkinter import *
import tkinter.messagebox

top = Tk()
top.geometry("200x400")

def mouse_click(event):
    print("click x=%d,y=%d"%(event.x,event.y))

L1 = Label(top,text="dev_id")
L1.pack()
dv1 = StringVar()
dv1.set('a')
E1 = Entry(top,bd = 5,textvariable=dv1)
E1.pack()

def hit_ok():
    tkinter.messagebox.showinfo("s1","s2")

canvas = Canvas(top,width=100,height=100)
canvas.bind("<Button-1>",mouse_click)
canvas.pack()
canvas.create_line(0,0,100,100)

B1 = Button(top,text="b1",command=hit_ok)
B1.pack()
top.mainloop()