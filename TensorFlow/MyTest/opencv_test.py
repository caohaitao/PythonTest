__author__ = 'ck_ch'
import cv2
import numpy as np

def picture_show():
    im = cv2.imread("test2.bmp")
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    h,w = gray.shape[:2]
    print(h)
    print(w)

    for i in range(h):
        for j in range(w):
            print(gray[i][j])

    cv2.imshow("image",gray)
    cv2.waitKey(0)

def video_show():
    cap = cv2.VideoCapture('E:\\MyLib\\opencv\\samples\\data\\tree.avi')
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        # cv2.imshow('frame',gray)
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def image_create():
    width = 200
    height = 200
    #image = np.zeros((height,width,3),dtype=np.uint8)
    image = np.full((height,width,3),fill_value=255,dtype=np.uint8)
    start_point=(10,10)
    end_point=(100,100)
    color=(255,0,0)
    line_width = 4
    line_type = 8
    cv2.line(image,start_point,end_point,color,line_width,line_type)

    x=30
    y=30
    rect_start=(x,y)
    x1 = 90
    y1 = 90
    rect_end=(x1,y1)
    cv2.rectangle(image,rect_start,rect_end,color,1,0)

    x = 100
    y = 100
    radius = 60
    circle_center=(x,y)
    cv2.circle(image,circle_center,radius,color)

    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(image, 'opencv', (10, 40), font, 1, (0, 0, 255), 1, False)

    cv2.imshow("img",image)
    cv2.waitKey(0)
    cv2.imwrite('out_put.jpg',image)

#picture_show()

#video_show()

image_create()