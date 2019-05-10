import cv2
from PIL import Image
import numpy as np

def trans_jpg_to_bin(jpg_path,out_path):
    img = Image.open(jpg_path)
    w = img.size[0]
    h = img.size[1]
    img = np.array(img)


    with open(out_path,'wb+') as f:
        bytes = w.to_bytes(4,byteorder='little')
        f.write(bytes)
        bytes = h.to_bytes(4,byteorder='little')
        f.write(bytes)
        f.write(img.tobytes())

def show_bmp_my(bmp_path):
    f = open(bmp_path,'rb')
    w=0
    h=0
    bytes = f.read(4)
    w = w.from_bytes(bytes,byteorder='little')
    bytes = f.read(4)
    h = h.from_bytes(bytes,byteorder='little')
    img = np.frombuffer(f.read(),dtype=np.uint8)
    f.close()
    img = img.reshape(h,w,3)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imshow('test',img)
    cv2.waitKey(0)


if __name__=='__main__':
    # trans_jpg_to_bin(
    #     r"F:\nnimages\coco\train2017\000000000009.jpg",
    #     r'e:\test.bmpmy'
    # )
    show_bmp_my(r'e:\test.bmpmy')