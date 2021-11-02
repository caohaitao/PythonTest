from PIL import Image,ImageDraw,ImageFont
import os
import cv2
import numpy as np

def trans_one_pic(from_path,to_path):
    image = Image.open(from_path)
    text = '老曹笔记 thetanet.cn'
    font = ImageFont.truetype('C:\Windows\Fonts\SIMYOU.TTF', 20)
    layer = image.convert('RGBA')
    text_overlay = Image.new('RGBA', layer.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)
    text_size_x, text_size_y = image_draw.textsize(text, font=font)
    # text_xy = (layer.size[0] - text_size_x,layer.size[1] - text_size_y)
    gap_x = (layer.size[0] - text_size_x) / 2
    text_xy = (gap_x, layer.size[1] / 2)
    image_draw.text(text_xy, text, font=font, fill=(255, 0, 0, 80))
    after = Image.alpha_composite(layer, text_overlay)
    after.save(to_path)

def trans_pics(from_dir,to_dir):
    datanames = os.listdir(from_dir)
    for name in datanames:
        s = os.path.splitext(name)
        if len(s)==1:
            continue
        if s[1] != '.jpg':
            continue
        fp = os.path.join(from_dir,name)
        tp = os.path.join(to_dir,s[0]+'.png')
        trans_one_pic(fp,tp)

if __name__=='__main__':
    # trans_pics(r'E:\work\几何算法图片',r'E:\work\logo_pics')


    # img = Image.open(r"E:\work\几何算法图片\cloud2d.jpg")
    # logo = Image.open(r"E:\code\web\DeepLookingWeb\images\logo-text.png")
    # logo = logo.resize((128, 32), Image.ANTIALIAS)
    # layer = Image.new('RGBA',img.size,(255,255,255,100))
    # xm = int((img.size[0] - logo.size[0])/2)
    # ym = int((img.size[1] - logo.size[1])/2)
    # layer.paste(logo,(xm,ym))
    # img_after = Image.composite(layer,img,layer)
    # img_after.show()


    # logo = Image.open(r"E:\code\web\DeepLookingWeb\images\logo-text.png")
    # logo = logo.resize((256, 64))
    # layer = Image.new('RGBA',(400,400),(255,255,255,0))
    # draw = ImageDraw.Draw(layer)
    # draw.bitmap((0, 0), logo,fill=(255,0,0))
    # layer.show()

    # img = Image.open(r"E:\work\几何算法图片\cloud2d.jpg")
    # logo = Image.open(r"E:\code\web\DeepLookingWeb\images\logo-text.png")
    # logo = logo.resize((256, 64))
    #
    # layer = Image.new('RGBA',img.size,(255,255,255,0))
    # img_after = Image.composite(img, logo, layer)
    # print(img_after.size)
    # fig = Image.blend(img,img_after,0.5)
    #
    # fig.show()

    # draw = ImageDraw.Draw(img)
    # xm = int((img.size[0] - 128)/2)
    # ym = int((img.size[1] - 32)/2)
    # draw.bitmap((xm,ym),logo)
    # img.show()



    # Read images
    src = cv2.imread(r"E:\code\web\DeepLookingWeb\images\logo-text.png")
    dst = cv2.imread(r"E:\work\几何算法图片\vs_dll_2.jpg")

    # Create a rough mask around the airplane.
    src_mask = np.zeros(src.shape, src.dtype)

    # 当然我们比较懒得话，就不需要下面两行，只是效果差一点。
    # 不使用的话我们得将上面一行改为 mask = 255 * np.ones(obj.shape, obj.dtype) <-- 全白
    # poly = np.array([[4, 80], [30, 54], [151, 63], [254, 37], [298, 90], [272, 134], [43, 122]], np.int32)
    # cv2.fillPoly(src_mask, [poly], (255, 255, 255))

    # 这是 飞机 CENTER 所在的地方
    center = (50, 50)

    # Clone seamlessly.
    output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
    cv2.imshow('img',output)
    cv2.waitKey(0)


    # 保存结果
    # cv2.imwrite("images/opencv-seamless-cloning-example.jpg", output);