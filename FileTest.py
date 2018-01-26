import struct
import numpy as np

file_name_map = {0:0}

if 2 in file_name_map.keys():
    file_name_map[2]+=1
else:
    file_name_map[2] = 0

a=1
b=2

WriteFileData = open('d:\\a.txt', 'wb')
bytes = a.to_bytes(1,byteorder='little')
WriteFileData.write(bytes)
bytes = b.to_bytes(1,byteorder='little')
r=WriteFileData.write(bytes)
r=WriteFileData.close()

WriteFileData = open('d:\\a.txt', 'rb')
bytes = WriteFileData.read(4)
b=b.from_bytes(bytes,byteorder="little")
bytes = WriteFileData.read(4)
a=a.from_bytes(bytes,byteorder="little")
WriteFileData.close()