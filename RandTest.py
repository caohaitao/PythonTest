import numpy as np

rs = np.random.rand(10000)
mrs={0:1}
a=mrs[0]
for r in rs:
    i = int(r*10)
    if i in mrs.keys():
        mrs[i] =int(mrs[i])+ 1
    else:
        mrs[i]=1

print(mrs)