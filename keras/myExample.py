import numpy as np

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(100)])

A = np.array([[1, 2, 3], [4, 5, 6]])
B = A.reshape((3,2))
print(x.shape)
print(y.shape)
print(A)
print(B)
print()

c = np.array([range(1,101), range(311,411), range(100)])
d = np.array([range(101,201), range(711,811), range(100)])

print(x.shape)

print()
c = np.transpose(c)
d = np.transpose(d)

print(c.shape)
print(d.shape)