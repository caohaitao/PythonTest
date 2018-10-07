__author__ = 'ck_ch'
import os

class picture_data:
    def __init__(self):
        self.file_name=''
        self.flag=0
    def print_data(self):
        print("file_name=%s,flag=%d" % (self.file_name,self.flag))

def get_picture_data():
    out_data=[]
    dir='data\\'
    for (root,dirs,files) in os.walk(dir):
        for item in files:
            d = os.path.join(root,item)
            pd = picture_data()
            pd.file_name = d
            pd.flag = ord(item[0])-65
            #pd.print_data()
            out_data.append(pd)

    return out_data

od = get_picture_data()

for o in od:
    o.print_data()




