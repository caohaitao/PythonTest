import struct
import numpy as np

a=1
b=2

WriteFileData = open('d:\\a.txt', 'wb')
bytes = a.to_bytes(4,byteorder='little')
WriteFileData.write(bytes)
bytes = b.to_bytes(4,byteorder='little')
r=WriteFileData.write(bytes)
r=WriteFileData.close()

WriteFileData = open('d:\\a.txt', 'rb')
bytes = WriteFileData.read(4)
b=b.from_bytes(bytes,byteorder="little")
bytes = WriteFileData.read(4)
a=a.from_bytes(bytes,byteorder="little")
WriteFileData.close()