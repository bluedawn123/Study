import numpy as np
import time
from numpy.random import rand

n = 150

a = np.array(rand[n, n])
b = np.array(rand[n, n])
c = np.array([[0] * n for _ in range(n)])


start = time.time()

for i in range(n):
    for j in range(n):
        for k in range(n):
            c[i][j] = a[i][k] * b[k][j]

print("파이썬 기능으로 계산한 결과 : , %.2f[sec]" % float(time.time() - start))

start = time.time()

c = np.dot(a, b)

print("Numpy를 사용하여 계산한 결과 : %.2f[sec]" % float(time.time() - start))



