import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1.데이터
a =  np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([item for item in subset])
    
    return np.array(aaa)

dataset = split_x(a, size)
print("=====================")
print(dataset)
print(dataset.shape)
print(type(dataset))

x = dataset[:, 0:4] #: 은 all, 즉 모든 행. 그리고 0:4는 0부터 4, 즉, 앞에 4개가 들어간다. 
y = dataset[:, 4]   #:은 역시 all, 그리고 4는 마지막 5번째만 가져오겠다는 의미.


