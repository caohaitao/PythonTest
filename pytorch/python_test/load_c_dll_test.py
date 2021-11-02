__author__ = 'ck_ch'
from ctypes import *
ll = cdll.LoadLibrary(r"E:\GameCode\WuZiQiHW\WuZiQiLib\x64\Release\MainDlgWZQ.dll")
#ll.DlgShow()
get_str = ll.get_str
get_str.restype = c_char_p
str = get_str()
print(str)