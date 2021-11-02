import cv2
def video_show():
    cap = cv2.VideoCapture(r"D:\eluosifangkuai.mp4")
    count = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',frame)

        pic_path = format(r"D:\video_pics\pic_%d.jpg"%count)
        cv2.imwrite(pic_path,frame)
        count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    video_show()