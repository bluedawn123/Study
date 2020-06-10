#39번 응용. 나누기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#데이터
'''
a =  np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        aaa.append([item for item in subset])
    
    return np.array(aaa)

z = split_x(a, size)
print("=====================")
print(z)

6행5열로 나뉜다. 
'''  
a = np.array(range(1, 11))
size = 5                   # time_steps = 5

x_predict = np.array([13,14,15,16])
#LSTM  모델을 완성하시오.

def split_x(seq, size):
    aaa = [] 
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = seq[i: (i+size-1)] #열 지정
        aaa.append([item for item in subset])
    
    return np.array(aaa)


def split_y(seq, size):
    bbb = [] 
    for i in range(len(seq) - size + 1):#seq - size +1  = 값 행종료값
        subset = [(i+size)] #열 지정
        bbb.append([item for item in subset])
    
    return np.array(bbb)

dataset1 = split_x(a, size)
print("=====================")
print(dataset1)


dataset2 = split_y(a, size)
print("=====================")
print(dataset2)

print()

