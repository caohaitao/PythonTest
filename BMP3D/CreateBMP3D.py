from PIL import Image
import cv2
import numpy as np
def img_show():
    img = Image.open(r"E:\tensorflow_datas\voc\voc2007\JPEGImages\000071.jpg")
    w = img.size[0]
    h = img.size[1]

    img = img.resize((200,200))
    print(w, h)
    img = np.array(img)

    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imshow('test',img)
    cv2.waitKey(0)
    fimg = img.astype(np.float32)
    f = open('gray.b','wb+')
    f.write(fimg.tobytes())
    f.close()

def img_read():
    with open('gray.b','rb') as f:
        img = np.frombuffer(f.read(),dtype=np.float32)
    img = img.astype(np.uint8)
    img = img.reshape(200,200)

    cv2.imshow('test',img)
    cv2.waitKey(0)

def trans_pic_from_gray_to_bmp32():
    dest = np.zeros((200,200*200),dtype=np.float32)
    with open('gray.b','rb') as f:
        dest[0] = np.frombuffer(f.read(),dtype=np.float32)

    f = open('test.bmp3d','wb+')
    d = 200
    h = 200
    w = 200
    bytes = d.to_bytes(4,byteorder='little')
    f.write(bytes)
    bytes = h.to_bytes(4,byteorder='little')
    f.write(bytes)
    bytes = w.to_bytes(4,byteorder='little')
    f.write(bytes)
    f.write(dest.tobytes())
    f.close()


if __name__ == '__main__':
    # img_show()
    # img_read()
    trans_pic_from_gray_to_bmp32()