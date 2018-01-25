import struct
import numpy as np

a=1.0
b=2.0


# 将long型写入到文件中
WriteFileData = open('d:\\a.txt', 'wb')
bytes = a.to_bytes(4)
WriteFileData.write(bytes)
bytes = b.to_bytes(4)
WriteFileData.write(bytes)
WriteFileData.close()